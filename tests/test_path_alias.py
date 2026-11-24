"""Path aliases: old paths keep reading the same series across a restructure.

``NodeScope.rename`` / ``NodeScope.move_to`` rewrite ``node.path`` for a whole
subtree. External callers that cached a path would otherwise lose their series
or, worse, silently start reading whichever unrelated node later claimed that
path. Those two mutators therefore record every rewritten path in
``energydb.node_path_alias``, and the *path-addressed read* routes (single
manifest ``path`` reads, edge-triple manifest reads, ``NodeScope.read`` by path,
``EdgeScope.read`` by triple) resolve through it:

* the UUID (and so the series data) stays the same;
* result ``path`` / ``from_path`` / ``to_path`` labels stay the *requested*
  old path;
* a path since claimed by a *different* node raises
  :class:`~energydb.errors.PathAliasConflictError` rather than picking one.

Write semantics and structural enumeration (``get`` / ``children`` /
``descendants`` / ``path`` / ``move_to(target)`` / ``create_node(parent=)``)
deliberately do NOT consult aliases: a renamed-away path is not a live address
for a write or a structural walk. Aliases are read-only to operate: only
``get_path_alias`` / ``list_path_aliases`` exist, no delete or re-point.

Live integration tests: skipped if ``TIMEDB_PG_DSN`` / ``TIMEDB_CH_URL``
aren't set.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import energydb as edb
import polars as pl
import pytest
from energydatamodel.reference import Reference
from energydb import Client
from energydb.errors import PathAliasConflictError, SeriesNotFoundError

if not (os.environ.get("TIMEDB_PG_DSN") and os.environ.get("TIMEDB_CH_URL")):
    pytest.skip(
        "TIMEDB_PG_DSN / TIMEDB_CH_URL not set: skipping path-alias tests",
        allow_module_level=True,
    )


@pytest.fixture
def client():
    c = Client()
    c.delete()
    c.create()
    yield c
    c.delete()
    c.close()


def _hours(n: int) -> list[datetime]:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    return [base + timedelta(hours=i) for i in range(n)]


def _register_power(client, scope, values=(3.5, 3.5)) -> None:
    scope.register_series(
        name="power",
        canonical_unit="MW",
        data_type="actual",
        timeseries_type="FLAT",
        retention="forever",
    )
    scope.write(
        pl.DataFrame({"valid_time": _hours(len(values)), "value": list(values)}),
        data_type="actual",
        name="power",
    )


def _power_manifest(path: str) -> pl.DataFrame:
    return pl.DataFrame({"path": [path], "data_type": ["actual"], "name": ["power"]})


# ---------------------------------------------------------------------------
# rename / move_to keep old paths readable, with the requested path label
# ---------------------------------------------------------------------------


def test_rename_keeps_old_path_reading_the_same_series(client):
    t01 = edb.wind.WindTurbine(name="T01", capacity=3.5)
    client.register_tree(edb.Portfolio(name="P", members=[edb.Site(name="S", members=[t01])]))
    _register_power(client, client.get_node("P", "S", "T01"))

    client.get_node("P", "S", "T01").rename("T01-A")

    # The old path is no longer a live node address...
    with pytest.raises(ValueError, match="Node not found"):
        client.get_node("P", "S", "T01").get()
    # ...but it still reads the same series, labelled with the requested path.
    out = client.get_node("P", "S", "T01").read(data_type="actual", name="power")
    assert out["value"].to_list() == [3.5, 3.5]
    by_path = client.get_node("P", "S", "T01").read(data_type="actual", name="power", output="by_path")
    assert list(by_path) == [edb.SeriesKey("P/S/T01", "actual", "power")]
    # And so does a manifest read through the same old path.
    read = client.read(_power_manifest("P/S/T01"))
    assert read["value"].to_list() == [3.5, 3.5]
    assert read["path"].unique().to_list() == ["P/S/T01"]

    alias = client.get_path_alias("P/S/T01")
    assert alias == {
        "path": "P/S/T01",
        "uuid": str(t01.id),
        "current_path": "P/S/T01-A",
        "is_alias": True,
    }


def test_move_to_keeps_old_subtree_paths_readable(client):
    t01 = edb.wind.WindTurbine(name="T01", capacity=3.5)
    client.register_tree(
        edb.Portfolio(
            name="P",
            members=[edb.Site(name="OldSite", members=[t01]), edb.Site(name="NewSite")],
        )
    )
    _register_power(client, client.get_node("P", "OldSite", "T01"))

    client.get_node("P", "OldSite", "T01").move_to(client.get_node("P", "NewSite"))

    out = client.get_node("P", "OldSite", "T01").read(data_type="actual", name="power")
    assert out["value"].to_list() == [3.5, 3.5]
    assert client.get_path_alias("P/OldSite/T01")["current_path"] == "P/NewSite/T01"
    # Moving one child out genuinely changes the old parent's subtree, so a
    # subtree read at the old parent is empty — only the moved node's own old
    # address keeps resolving.
    assert client.get_node("P", "OldSite").read(data_type="actual", name="power", output="by_path") == {}


def test_rename_records_whole_subtree_and_chained_renames(client):
    client.register_tree(
        edb.Portfolio(name="P", members=[edb.Site(name="A", members=[edb.wind.WindTurbine(name="T01", capacity=3.5)])])
    )
    _register_power(client, client.get_node("P", "A", "T01"))

    client.get_node("P", "A").rename("B")
    # Both the renamed root and its descendant kept their pre-rename path.
    assert client.get_path_alias("P/A")["current_path"] == "P/B"
    assert client.get_path_alias("P/A/T01")["current_path"] == "P/B/T01"

    client.get_node("P", "B").rename("C")
    # The newest old path and the original both still resolve to the node.
    assert client.get_path_alias("P/B")["current_path"] == "P/C"
    assert client.get_path_alias("P/A")["current_path"] == "P/C"
    assert client.get_node("P", "A", "T01").read(data_type="actual", name="power")["value"].to_list() == [3.5, 3.5]


def test_edge_triple_old_endpoint_paths_still_read(client):
    bus_a = edb.grid.JunctionPoint(name="BusA")
    bus_b = edb.grid.JunctionPoint(name="BusB")
    client.register_tree(edb.Portfolio(name="Grid", members=[bus_a, bus_b]))
    line = edb.grid.Line(
        name="L1",
        capacity=500,
        from_element=Reference(bus_a),
        to_element=Reference(bus_b),
        timeseries=[edb.TimeSeries(name="flow", unit="MW", data_type=edb.DataType.ACTUAL)],
    )
    edge_uuid = client.create_edge(line)
    client.get_edge(uuid=edge_uuid).write(
        pl.DataFrame({"valid_time": _hours(2), "value": [1.0, 2.0]}), data_type="actual", name="flow"
    )

    client.get_node("Grid", "BusA").rename("BusA2")

    old = client.get_edge(("Grid", "BusA"), ("Grid", "BusB"), type="Line").read(data_type="actual", name="flow")
    assert old["value"].to_list() == [1.0, 2.0]

    manifest = pl.DataFrame(
        {
            "from_path": ["Grid/BusA"],
            "to_path": ["Grid/BusB"],
            "edge_type": ["Line"],
            "data_type": ["actual"],
            "name": ["flow"],
        }
    )
    read = client.read(manifest)
    assert read["value"].to_list() == [1.0, 2.0]
    assert read["from_path"].unique().to_list() == ["Grid/BusA"]
    key = next(iter(client.read(manifest, output="by_path")))
    assert (key.from_path, key.to_path) == ("Grid/BusA", "Grid/BusB")


# ---------------------------------------------------------------------------
# Conflicts: an old path since claimed by a different node
# ---------------------------------------------------------------------------


def test_old_path_taken_by_another_node_raises_conflict(client):
    client.register_tree(
        edb.Portfolio(
            name="P",
            members=[edb.Site(name="A", members=[edb.Site(name="X")]), edb.Site(name="Other")],
        )
    )
    _register_power(client, client.get_node("P", "A", "X"))
    client.get_node("P", "A", "X").rename("Y")
    # A *different* node now holds the old path.
    client.create_node(node_type="Site", name="X", parent=("P", "A"))

    with pytest.raises(PathAliasConflictError) as excinfo:
        client.get_node("P", "A", "X").read(data_type="actual", name="power")

    err = excinfo.value
    assert err.path == "P/A/X"
    assert len(err.candidates) == 2
    assert {c["current_path"] for c in err.candidates} == {"P/A/X", "P/A/Y"}
    # ...and never the other node's data: that node has no series at all.
    with pytest.raises(PathAliasConflictError):
        client.read(_power_manifest("P/A/X"))
    with pytest.raises(PathAliasConflictError):
        client.get_path_alias("P/A/X")


def test_conflict_raises_even_under_on_missing_skip(client):
    """A conflict is an addressing error, not a catalog gap: skipping missing
    series must not turn "two nodes claim this path" into an empty result."""
    client.register_tree(edb.Portfolio(name="P", members=[edb.Site(name="A", members=[edb.Site(name="X")])]))
    client.get_node("P", "A", "X").rename("Y")
    client.create_node(node_type="Site", name="X", parent=("P", "A"))

    manifest = _power_manifest("P/A/X")
    with pytest.raises(PathAliasConflictError):
        client.read(manifest, on_missing="skip")


def test_endpoint_conflict_without_any_edge_raises(client):
    """The conflict must surface even when no edge joins on the conflicted
    endpoint: an edge-only join would report it as a missing edge instead."""
    client.register_tree(edb.Portfolio(name="P", members=[edb.Site(name="B"), edb.Site(name="C")]))
    client.get_node("P", "B").rename("B2")
    client.create_node(node_type="Site", name="B", parent=("P",))

    manifest = pl.DataFrame(
        {
            "from_path": ["P/B"],
            "to_path": ["P/C"],
            "edge_type": ["Line"],
            "data_type": ["actual"],
            "name": ["flow"],
        }
    )
    with pytest.raises(PathAliasConflictError):
        client.read(manifest)
    with pytest.raises(PathAliasConflictError):
        client.get_edge(("P", "B"), ("P", "C"), type="Line").read(data_type="actual", name="flow")


# ---------------------------------------------------------------------------
# Writes and structural enumeration are untouched
# ---------------------------------------------------------------------------


def test_writes_and_structural_enumeration_do_not_follow_aliases(client):
    client.register_tree(
        edb.Portfolio(name="P", members=[edb.Site(name="A", members=[edb.Site(name="X")]), edb.Site(name="B")])
    )
    _register_power(client, client.get_node("P", "A", "X"))
    client.get_node("P", "A", "X").rename("Y")

    # Writing through the renamed-away path must not silently land anywhere.
    write = pl.DataFrame(
        {
            "path": ["P/A/X"],
            "data_type": ["actual"],
            "name": ["power"],
            "valid_time": _hours(1),
            "value": [9.0],
        }
    )
    with pytest.raises(SeriesNotFoundError):
        client.write(write)

    # Structural navigation keeps using live paths only.
    with pytest.raises(ValueError, match="Node not found"):
        client.get_node("P", "A", "X").get()
    assert [c["name"] for c in client.get_node("P", "A").children()] == ["Y"]
    assert client.get_node("P", "A", "X").resolve(data_type="actual") is not None  # read-resolve does follow


# ---------------------------------------------------------------------------
# Lifecycle: namespace isolation, cascade on delete, dry-run / txn rollback
# ---------------------------------------------------------------------------


def test_alias_cascades_with_its_node(client):
    client.register_tree(edb.Portfolio(name="P", members=[edb.Site(name="A", members=[edb.Site(name="X")])]))
    client.get_node("P", "A", "X").rename("Y")
    assert client.get_path_alias("P/A/X") is not None

    client.get_node("P", "A", "Y").delete()
    assert client.get_path_alias("P/A/X") is None
    assert client.list_path_aliases() == []


def test_alias_listing_filters_and_pagination(client):
    client.register_tree(
        edb.Portfolio(
            name="P",
            members=[
                edb.Site(name="A", members=[edb.Site(name="X"), edb.Site(name="Y")]),
                edb.Site(name="B"),
            ],
        )
    )
    client.get_node("P", "A").rename("A2")

    listed = client.list_path_aliases()
    assert {r["path"] for r in listed} == {"P/A", "P/A/X", "P/A/Y"}
    assert all(r["current_path"].startswith("P/A2") for r in listed)
    assert all(r["node_type"] == "Site" for r in listed)

    prefixed = client.list_path_aliases(prefix="P/A/X")
    assert [r["path"] for r in prefixed] == ["P/A/X"]

    page = client.list_path_aliases(limit=2)
    assert len(page) == 2
    assert [r["path"] for r in client.list_path_aliases(after=(page[-1]["path"], page[-1]["uuid"]))] == ["P/A/Y"]


def test_alias_is_namespace_isolated(client):
    view_a = client.namespace("alias-ns-a")
    view_b = client.namespace("alias-ns-b")
    view_a.register_tree(edb.Portfolio(name="P", members=[edb.Site(name="A")]))
    view_a.get_node("P", "A").rename("B")

    # The alias belongs to ns-a only, and ns-b's identically named live path is
    # not a conflict for it.
    assert view_a.get_path_alias("P/A")["current_path"] == "P/B"
    assert view_b.get_path_alias("P/A") is None
    assert view_b.list_path_aliases() == []
    assert [r["path"] for r in view_a.list_path_aliases()] == ["P/A"]


def test_dry_run_and_rollback_leave_no_alias(client):
    client.register_tree(edb.Portfolio(name="P", members=[edb.Site(name="A")]))

    diff = client.get_node("P", "A").rename("B", dry_run=True)
    assert diff is not None
    assert client.list_path_aliases() == []
    assert client.get_node("P", "A").get().name == "A"

    with pytest.raises(RuntimeError):
        with client.transaction() as txn:
            txn.get_node("P", "A").rename("B")
            raise RuntimeError("abort")
    assert client.list_path_aliases() == []
    assert client.get_node("P", "A").get().name == "A"

    with client.transaction() as txn:
        txn.get_node("P", "A").rename("B")
        txn.commit()
    assert [r["path"] for r in client.list_path_aliases()] == ["P/A"]


def test_engine_and_sequential_reads_agree_on_an_aliased_path(client):
    """The ClickHouse meta-engine only knows live paths, so an alias-resolved
    read must not take it: sequential and engine-enabled sessions return the
    same values."""
    client.register_tree(edb.Portfolio(name="P", members=[edb.Site(name="A", members=[edb.Site(name="X")])]))
    _register_power(client, client.get_node("P", "A", "X"), values=(1.0, 2.0, 3.0))
    client.get_node("P", "A", "X").rename("Y")

    from energydb.client import AsyncClient

    async def read_with_engine(disabled: bool):
        root = AsyncClient(pg_conninfo=os.environ["TIMEDB_PG_DSN"], ch_url=os.environ["TIMEDB_CH_URL"])
        await root.open()
        if disabled:
            root._engine_unavailable = True
        try:
            return await root.get_node("P", "A", "X").read(data_type="actual", name="power")
        finally:
            await root.close()

    import asyncio

    engine_read = asyncio.run(read_with_engine(disabled=False))
    sequential_read = asyncio.run(read_with_engine(disabled=True))
    assert engine_read["value"].to_list() == [1.0, 2.0, 3.0]
    assert engine_read["value"].to_list() == sequential_read["value"].to_list()
