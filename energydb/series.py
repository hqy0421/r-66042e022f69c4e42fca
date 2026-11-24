"""Series-table operations: register, resolve for read.

The series table is owned by exactly one of ``node_uuid`` / ``edge_uuid``
(DB CHECK enforces). Internal APIs use a single ``(owner_col, owner_uuid)``
shape: callers always know which side they're on, so encoding "set one, leave
the other None" buys nothing.

``series_id BIGINT`` stays as the timedb-internal handle; it never leaves the
energydb / timedb pair.

Retention tier names are owned by timedb (see :data:`timedb.RETENTION_TIERS`).
energydb consumes the set as a runtime guard but does not encode the values
into its PG schema, so adding a tier in timedb does not require an energydb
migration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal
from uuid import UUID

import polars as pl
from timedb import RETENTION_TIERS

from energydb.errors import AlreadyExistsError, EdgeNotFoundError, ValidationError
from energydb.models import SQL_SCHEMA_PREFIX as P
from energydb.paths import (
    _candidate_cte,
    _conflict_claimants,
    _path_alias_conflict,
    ambiguous_edge_error,
    edge_address_repr,
)

OwnerCol = Literal["node_uuid", "edge_uuid"]

_VALID_TIMESERIES_TYPES = {"FLAT", "OVERLAPPING"}

# Defaults are picked by series shape: actuals (FLAT) should never expire,
# forecasts (OVERLAPPING) age out under the medium tier.
_DEFAULT_RETENTION_BY_SHAPE = {
    "FLAT": "forever",
    "OVERLAPPING": "medium",
}

_CONFLICT_CONSTRAINT_BY_OWNER = {
    "node_uuid": "series_node_uniq",
    "edge_uuid": "series_edge_uniq",
}


def _validate_timeseries_type(ts_type: str) -> None:
    if ts_type not in _VALID_TIMESERIES_TYPES:
        raise ValidationError(f"Unknown timeseries_type {ts_type!r}. Valid values: {sorted(_VALID_TIMESERIES_TYPES)}")


def _validate_retention(retention: str) -> None:
    if retention not in RETENTION_TIERS:
        raise ValidationError(f"Unknown retention {retention!r}. Valid values: {sorted(RETENTION_TIERS)}")


def validate_name(name: str, *, kind: str) -> None:
    """Reject empty names or names containing ``/``.

    ``/`` is the path separator on the read API (``path: pl.Utf8`` joined
    with ``/``), so allowing it inside a node/edge/series name would make
    the joined path ambiguous. The same constraint is enforced at the PG
    level by ``CheckConstraint("name !~ '/' AND length(name) > 0")``.
    """
    if not isinstance(name, str):
        raise TypeError(f"{kind} name must be a string, got {type(name).__name__}")
    if len(name) == 0:
        raise ValidationError(f"{kind} name must be non-empty.")
    if "/" in name:
        raise ValidationError(
            f"{kind} name {name!r} contains '/'. '/' is reserved as the path "
            f"separator and may not appear inside a node, edge, or series name."
        )


# Column order of SERIES_INSERT_SQL params: shared by the single-row upsert
# below and the batched register_tree path in _persist.
SERIES_INSERT_COLUMNS = (
    "node_uuid",
    "edge_uuid",
    "data_type",
    "name",
    "canonical_unit",
    "timeseries_type",
    "retention",
    "description",
)


def prepare_series_row(
    *,
    owner_col: OwnerCol,
    owner_uuid: UUID,
    data_type: str,
    name: str,
    canonical_unit: str,
    timeseries_type: str,
    retention: str | None = None,
    description: str | None = None,
) -> tuple:
    """Validate + normalize one series declaration; return the INSERT param tuple.

    Pure (no I/O): validation and the shape-derived retention default
    (FLAT → ``"forever"``, OVERLAPPING → ``"medium"``) live here so the
    single-row upsert and the batched ``register_tree`` insert agree exactly.
    Param order is :data:`SERIES_INSERT_COLUMNS`.
    """
    _validate_timeseries_type(timeseries_type)
    validate_name(name, kind="series")
    if retention is None:
        retention = _DEFAULT_RETENTION_BY_SHAPE[timeseries_type]
    _validate_retention(retention)

    node_uuid = owner_uuid if owner_col == "node_uuid" else None
    edge_uuid = owner_uuid if owner_col == "edge_uuid" else None
    return (node_uuid, edge_uuid, data_type, name, canonical_unit, timeseries_type, retention, description)


async def register_series(
    conn,
    *,
    owner_col: OwnerCol,
    owner_uuid: UUID,
    data_type: str,
    name: str,
    canonical_unit: str,
    timeseries_type: str,
    retention: str | None = None,
    description: str | None = None,
) -> int:
    """Insert a new series row owned by ``owner_uuid``; return its series_id.

    ``owner_col`` is one of ``"node_uuid"`` / ``"edge_uuid"``. The DB
    CHECK enforces that exactly one of those columns is non-null on each
    row. ``retention``, ``canonical_unit``, and the owner are immutable
    after insert (DB trigger enforces). ``timeseries_type`` is mutable.

    If ``retention`` is omitted, it is derived from ``timeseries_type``:
    FLAT (actuals) → ``"forever"``, OVERLAPPING (forecasts) → ``"medium"``.
    """
    params = prepare_series_row(
        owner_col=owner_col,
        owner_uuid=owner_uuid,
        data_type=data_type,
        name=name,
        canonical_unit=canonical_unit,
        timeseries_type=timeseries_type,
        retention=retention,
        description=description,
    )
    retention = params[SERIES_INSERT_COLUMNS.index("retention")]
    conflict_constraint = _CONFLICT_CONSTRAINT_BY_OWNER[owner_col]

    row = await (
        await conn.execute(
            f"""
            INSERT INTO {P}series
                (node_uuid, edge_uuid, data_type, name, canonical_unit,
                 timeseries_type, retention, description)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT ON CONSTRAINT {conflict_constraint} DO NOTHING
            RETURNING series_id
            """,
            params,
        )
    ).fetchone()

    if row is not None:
        return row[0]

    existing = await (
        await conn.execute(
            f"SELECT series_id, canonical_unit, retention "
            f"FROM {P}series "
            f"WHERE {owner_col} = %s AND data_type = %s AND name = %s",
            (owner_uuid, data_type, name),
        )
    ).fetchone()
    if existing is None:
        raise RuntimeError("Insert conflict but no existing row found: concurrency bug")
    existing_sid, existing_unit, existing_retention = existing
    if existing_unit != canonical_unit or existing_retention != retention:
        raise AlreadyExistsError(
            f"Series ({owner_col}={owner_uuid}, data_type={data_type!r}, name={name!r}) "
            f"already exists with canonical_unit={existing_unit!r}, "
            f"retention={existing_retention!r}; cannot re-register with "
            f"canonical_unit={canonical_unit!r}, retention={retention!r}. "
            f"These fields are immutable, register a new series instead."
        )
    return existing_sid


_SUBTREE_META_SCHEMA = {
    "series_id": pl.Int64,
    "canonical_unit": pl.Utf8,
    "timeseries_type": pl.Utf8,
    "retention": pl.Utf8,
    "node_uuid": pl.Utf8,
    "data_type": pl.Utf8,
    "name": pl.Utf8,
    "path": pl.Utf8,
}


def _subtree_frame(rows: Sequence[tuple], *, with_alias_row: bool) -> pl.DataFrame:
    """Build the per-series subtree read-meta frame from raw rows.

    ``with_alias_row`` is for the alias-aware branch, whose rows carry two
    trailing columns: ``_alias`` (the prefix was answered by an alias, so the
    meta-engine's live-path predicate cannot be trusted) and ``n_cand`` (a
    caller-side conflict check). A LEFT-JOIN row for a series-less candidate is
    dropped here rather than packaged as a bogus series.
    """
    data: list[dict[str, Any]] = []
    aliased = False
    for row in rows:
        if with_alias_row:
            aliased = aliased or bool(row[8])
            if row[0] is None:  # series-less candidate (LEFT JOIN row)
                continue
            row = row[:8]
        data.append(
            {
                "series_id": row[0],
                "canonical_unit": row[1],
                "timeseries_type": row[2],
                "retention": row[3],
                "node_uuid": row[4],
                "data_type": row[5],
                "name": row[6],
                "path": row[7],
            }
        )
    df = pl.DataFrame(data, schema=_SUBTREE_META_SCHEMA)
    if with_alias_row:
        # Attribute the read as alias-resolved so execute_read can skip the
        # engine leg (see energydb._io.execute_read).
        df = df.with_columns(pl.lit(aliased).alias("_alias"))
    return df


async def resolve_subtree_series_for_read(
    conn,
    *,
    root_path: str | None = None,
    start_uuid: UUID | None = None,
    rel_path: str | None = None,
    where_conds: list[str] | None = None,
    where_params: list[Any] | None = None,
    data_type: str | None = None,
    name: str | None = None,
    path_aliases: bool = False,
) -> pl.DataFrame:
    """Resolve a node subtree straight to per-series read meta in ONE round-trip.

    A single query walks root → subtree (materialized-path prefix scan) →
    ``series`` ⋈ ``node``, replacing the former ``resolve_node_uuid`` +
    ``resolve_subtree_uuids`` + per-owner series scan chain on the node
    read path (which round-tripped the resolved uuid set back to PG as an
    ``ANY()`` param). The root is given either as ``root_path`` (a ``/``-joined
    path the caller already holds) or as ``start_uuid`` (+ optional ``rel_path``
    for relative navigation), whose path is derived inline via a CTE so the
    resolve stays one round-trip.

    ``where_conds`` / ``where_params`` are extra subtree-node predicates from
    :func:`build_filter_conditions` (qualified with ``table_alias="n"``).
    Returns one row per series with the per-series read-meta columns
    (``series_id``, ``canonical_unit``, ``timeseries_type``, ``retention``,
    ``node_uuid``, ``data_type``, ``name``, ``path``). Empty df if the subtree
    is empty or nothing matches.

    ``path_aliases=True`` (read pipelines, ``root_path`` only) additionally
    consults :class:`~energydb.models.NodePathAlias`: the requested prefix may
    name a path the root node has been renamed/moved away from. The returned
    ``path`` then stays the *requested* prefix plus each series' relative
    suffix, so a caller's old path keys keep matching. The frame gains an
    ``_alias`` bool column (True when the prefix came from an alias) for the
    read pipeline's engine guard, and the resolve raises
    :class:`~energydb.errors.PathAliasConflictError` when the prefix is claimed
    by more than one node. The ``start_uuid`` forms are identity-addressed and
    ignore the flag.
    """
    cte_params: list[Any] = []
    if root_path is not None:
        root_cte = "SELECT %s::text AS rp"
        cte_params.append(root_path)
    elif start_uuid is not None and rel_path:
        root_cte = f"SELECT (path || '/' || %s) AS rp FROM {P}node WHERE uuid = %s"
        cte_params.extend([rel_path, start_uuid])
    elif start_uuid is not None:
        root_cte = f"SELECT path AS rp FROM {P}node WHERE uuid = %s"
        cte_params.append(start_uuid)
    else:
        raise ValidationError("resolve_subtree_series_for_read needs root_path or start_uuid.")

    where_parts = list(where_conds or [])
    where_vals: list[Any] = list(where_params or [])
    if data_type:
        where_parts.append("s.data_type = %s")
        where_vals.append(data_type)
    if name:
        where_parts.append("s.name = %s")
        where_vals.append(name)
    where_clause = (" AND " + " AND ".join(where_parts)) if where_parts else ""

    # The prefix can be DB-derived, so it cannot always be escaped Python-side;
    # escaping LIKE metacharacters in SQL keeps this to one round-trip.
    if path_aliases and root_path is not None:
        # The requested prefix may be a path the root moved away from. Resolve
        # it to a node through the alias table, then run the usual prefix scan
        # from that node's *current* path — but label rows with the requested
        # prefix so a caller's old keys keep matching.
        #
        # series is LEFT-JOINed (filters in the ON clause) so a candidate with
        # no matching series still produces a row: that is what lets the
        # claimant count be read even from an empty resolve, keeping a conflict
        # between two series-less nodes from silently reading as "no data".
        join_extra = (" AND " + " AND ".join(where_parts)) if where_parts else ""
        sql = rf"""
        WITH req AS (SELECT %s::text AS req_rp),
        cand AS (
            SELECT n.uuid, n.path AS cur_rp
            FROM req r JOIN {P}node n ON n.path = r.req_rp
            UNION ALL
            SELECT a.node_uuid, nn.path
            FROM req r JOIN {P}node_path_alias a ON a.old_path = r.req_rp
                        JOIN {P}node nn ON nn.uuid = a.node_uuid
        ), root AS (
            SELECT r.req_rp, (SELECT count(DISTINCT uuid) FROM cand) AS n_cand,
                   (SELECT min(cur_rp) FROM cand) AS cur_rp
            FROM req r
        )
        SELECT s.series_id, s.canonical_unit, s.timeseries_type, s.retention,
               s.node_uuid::text AS node_uuid, s.data_type, s.name,
               %s::text || substring(n.path FROM length(r.cur_rp) + 1) AS path,
               (r.cur_rp <> r.req_rp) AS _alias, r.n_cand
        FROM root r
        JOIN {P}node n
          ON (n.path = r.cur_rp
              OR n.path LIKE
                 replace(replace(replace(r.cur_rp, E'\\', E'\\\\'), '%%', E'\\%%'), '_', E'\\_')
                 || '/%%' ESCAPE '\')
        LEFT JOIN {P}series s ON s.node_uuid = n.uuid{join_extra}
        """
        rows = await (await conn.execute(sql, (cte_params[0], root_path, *where_vals))).fetchall()
        if rows and rows[0][9] > 1:
            raise _path_alias_conflict(await _conflict_claimants(conn, [root_path]))
        return _subtree_frame(rows, with_alias_row=True)

    sql = rf"""
        WITH root AS ({root_cte})
        SELECT s.series_id, s.canonical_unit, s.timeseries_type, s.retention,
               s.node_uuid::text AS node_uuid, s.data_type, s.name, n.path AS path
        FROM root r
        JOIN {P}node n
          ON (n.path = r.rp
              OR n.path LIKE
                 replace(replace(replace(r.rp, E'\\', E'\\\\'), '%%', E'\\%%'), '_', E'\\_')
                 || '/%%' ESCAPE '\')
        JOIN {P}series s ON s.node_uuid = n.uuid
        WHERE TRUE{where_clause}
    """
    rows = await (await conn.execute(sql, [*cte_params, *where_vals])).fetchall()
    return _subtree_frame(rows, with_alias_row=False)


async def resolve_edge_series_for_read(
    conn,
    *,
    edge_uuid: UUID | None = None,
    from_path: str | None = None,
    to_path: str | None = None,
    edge_type: str | None = None,
    edge_name: str | None = None,
    data_type: str | None = None,
    name: str | None = None,
    path_aliases: bool = False,
) -> pl.DataFrame:
    """Resolve an edge's series for a read in ONE round-trip.

    The edge is addressed by ``edge_uuid`` OR by its
    ``(from_path, to_path, edge_type)`` triple, optionally narrowed by
    ``edge_name``; the triple form collapses the former three-step chain
    (paths → uuids, edge lookup, series scan) into a single query. The
    endpoint paths, ``edge_type`` and ``edge_name`` ride along on the join, so
    the post-read attach step needs no further PG calls.

    ``name`` / ``data_type`` filter the *series*; ``edge_name`` identifies the
    *edge*, the two names are deliberately separate.

    ``series`` is LEFT-JOINed so an existing edge with no (matching) series
    still returns a row, distinguishing "edge not found" from "no series":

    * triple-addressed + edge missing → raises
      :class:`~energydb.errors.EdgeNotFoundError` (the same contract as a
      standalone ``resolve_edge_uuid`` lookup);
    * triple-addressed + several parallel edges matched → raises
      :class:`~energydb.errors.AmbiguousEdgeError` (pass ``edge_name``);
    * uuid-addressed + edge missing → empty df, by contract;
    * edge exists, nothing matches → empty df.

    ``path_aliases=True`` (read pipelines) additionally resolves each endpoint
    path through :class:`~energydb.models.NodePathAlias`, so a triple naming a
    path an endpoint has since moved away from still finds the edge. The
    returned ``from_path`` / ``to_path`` stay the *requested* paths, an
    ``_alias`` bool column reports whether an endpoint came from an alias (the
    read pipeline's engine guard), and an endpoint path claimed by several
    nodes raises :class:`~energydb.errors.PathAliasConflictError` — even when
    no edge joins on it.
    """
    join_conds = ["s.edge_uuid = e.uuid"]
    join_params: list[Any] = []
    if data_type:
        join_conds.append("s.data_type = %s")
        join_params.append(data_type)
    if name:
        join_conds.append("s.name = %s")
        join_params.append(name)

    if edge_uuid is not None:
        where_sql = "e.uuid = %s"
        where_params: list[Any] = [edge_uuid]
    elif from_path is not None and to_path is not None and edge_type is not None:
        where_sql = "fn.path = %s AND tn.path = %s AND e.edge_type = %s"
        where_params = [from_path, to_path, edge_type]
        if edge_name is not None:
            where_sql += " AND e.name = %s"
            where_params.append(edge_name)
    else:
        raise ValidationError("resolve_edge_series_for_read needs edge_uuid or (from_path, to_path, edge_type).")

    alias_triple = path_aliases and edge_uuid is None

    # ::text skips psycopg's per-row UUID parse. The edge uuid comes from e, not
    # s, so it is populated on the LEFT-JOIN row of a series-less edge, which is
    # the row the ambiguity check counts.
    if alias_triple:
        assert from_path is not None and to_path is not None and edge_type is not None
        # Endpoint paths resolve live-or-alias (as on the node route), and the
        # projected paths stay the requested ones. Two marker branches report an
        # endpoint path claimed by several nodes even when no edge joins on it,
        # which the edge ⋈ candidate join alone would read as "edge not found".
        # Row layout: (series_id, canonical_unit, timeseries_type, retention,
        # edge_uuid, data_type, name, edge_type, edge_name, from_path, to_path,
        # conflict_side, f_alias, t_alias).
        T = "NULL::text"
        B = "NULL::bool"
        NF, NT = "NULL::bigint", "NULL::text"
        FILL_AFTER_SID = [NF] + [NT] * 8  # cols 0..8 on a from_path marker
        FILL_TO_FROM = [NF] + [NT] * 9  # cols 0..9 on a to_path marker
        edge_where = "e.edge_type = %s" + (" AND e.name = %s" if edge_name is not None else "")
        edge_params = [edge_type, *([edge_name] if edge_name is not None else [])]
        sql = (
            _candidate_cte("f", leading=True)
            + _candidate_cte("t", leading=False)
            + f"""
        SELECT s.series_id, s.canonical_unit, s.timeseries_type, s.retention,
               e.uuid::text, s.data_type, s.name, e.edge_type, e.name,
               cf.req_path AS from_path, ct.req_path AS to_path,
               {T} AS conflict_side,
               (cf.cur_path <> cf.req_path) AS f_alias,
               (ct.cur_path <> ct.req_path) AS t_alias
        FROM {P}edge e
        JOIN c_f cf ON cf.uuid = e.from_node_uuid
        JOIN c_t ct ON ct.uuid = e.to_node_uuid
        LEFT JOIN {P}series s ON """
            + " AND ".join(join_conds)
            + f" WHERE {edge_where}"
            + f"""
        UNION ALL
        SELECT {", ".join(FILL_AFTER_SID)}, x.req_path, {T}, 'from_path', {B}, {B}
        FROM (SELECT req_path FROM c_f GROUP BY req_path HAVING count(DISTINCT uuid) > 1) x
        UNION ALL
        SELECT {", ".join(FILL_TO_FROM)}, y.req_path, 'to_path', {B}, {B}
        FROM (SELECT req_path FROM c_t GROUP BY req_path HAVING count(DISTINCT uuid) > 1) y
        """
        )
        rows = await (
            await conn.execute(sql, ([from_path], [to_path], *join_params, *edge_params))
        ).fetchall()

        conflicted: dict[str, dict[str, str]] = {}
        for row in rows:
            if row[11] == "from_path":
                conflicted.setdefault(row[9], {})
            elif row[11] == "to_path":
                conflicted.setdefault(row[10], {})
        if conflicted:
            raise _path_alias_conflict(await _conflict_claimants(conn, sorted(conflicted)))

        if not rows:
            raise EdgeNotFoundError(
                f"Edge not found: {edge_address_repr(from_path, to_path, edge_type, edge_name)}",
                from_path=from_path,
                to_path=to_path,
                edge_type=edge_type,
                name=edge_name,
            )
        matched = {r[4]: r[8] for r in rows}
        if len(matched) > 1:
            raise ambiguous_edge_error(
                from_path=from_path,
                to_path=to_path,
                edge_type=edge_type,
                matches=list(matched.items()),
                fix="pass name= to address one of them",
            )
        aliased = any(bool(r[12]) or bool(r[13]) for r in rows)
        frame = _edge_series_frame(rows)
        return frame.with_columns(pl.lit(aliased).alias("_alias"))

    sql = (
        "SELECT s.series_id, s.canonical_unit, s.timeseries_type, s.retention, "
        "e.uuid::text, s.data_type, s.name, "
        "e.edge_type AS edge_type, e.name AS edge_name, fn.path AS from_path, tn.path AS to_path "
        f"FROM {P}edge e "
        f"JOIN {P}node fn ON fn.uuid = e.from_node_uuid "
        f"JOIN {P}node tn ON tn.uuid = e.to_node_uuid "
        f"LEFT JOIN {P}series s ON " + " AND ".join(join_conds) + " "
        "WHERE " + where_sql
    )
    rows = await (await conn.execute(sql, [*join_params, *where_params])).fetchall()

    if edge_uuid is None:
        assert from_path is not None and to_path is not None and edge_type is not None
        if not rows:
            raise EdgeNotFoundError(
                f"Edge not found: {edge_address_repr(from_path, to_path, edge_type, edge_name)}",
                from_path=from_path,
                to_path=to_path,
                edge_type=edge_type,
                name=edge_name,
            )
        # One row per (edge, matching series): several edges is ambiguous,
        # several series on one edge is ordinary.
        matched = {r[4]: r[8] for r in rows}
        if len(matched) > 1:
            raise ambiguous_edge_error(
                from_path=from_path,
                to_path=to_path,
                edge_type=edge_type,
                matches=list(matched.items()),
                fix="pass name= to address one of them",
            )

    return _edge_series_frame(rows)


def _edge_series_frame(rows: Sequence[tuple]) -> pl.DataFrame:
    """Build the per-series edge read-meta frame from raw resolver rows.

    Only the first eleven columns are read (the alias branch appends conflict /
    alias markers after them); a LEFT-JOIN row for a series-less edge is
    dropped rather than packaged as a bogus series.
    """
    return pl.DataFrame(
        [
            {
                "series_id": r[0],
                "canonical_unit": r[1],
                "timeseries_type": r[2],
                "retention": r[3],
                "edge_uuid": r[4],
                "data_type": r[5],
                "name": r[6],
                "edge_type": r[7],
                "edge_name": r[8],
                "from_path": r[9],
                "to_path": r[10],
            }
            for r in rows
            if r[0] is not None  # LEFT-JOIN row for a series-less edge
        ],
        schema={
            "series_id": pl.Int64,
            "canonical_unit": pl.Utf8,
            "timeseries_type": pl.Utf8,
            "retention": pl.Utf8,
            "edge_uuid": pl.Utf8,
            "data_type": pl.Utf8,
            "name": pl.Utf8,
            "edge_type": pl.Utf8,
            "edge_name": pl.Utf8,
            "from_path": pl.Utf8,
            "to_path": pl.Utf8,
        },
    )
