"""端到端自测：基于 ASGI transport 同时跑调度器与两台节点模拟器。

运行：
  PYTHONPATH=/tmp/pylibs PLATFORM_DB=/tmp/platform-test/test.db \
  python3 -m pytest tests/ -v
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid

import httpx
import pytest

# 必须在导入 app 之前配置环境
TEST_DIR = f"/tmp/platform-test-{uuid.uuid4().hex[:8]}"
os.environ["PLATFORM_DB"] = os.path.join(TEST_DIR, "test.db")
os.environ["PLATFORM_NODES"] = (
    "node-a|模拟节点A|http://testnode-a:9001|8192,"
    "node-b|模拟节点B|http://testnode-b:9002|8192"
)
os.environ["NODE_CALL_TIMEOUT"] = "1.0"
os.environ["LOAD_TIMEOUT"] = "4"
os.environ["UNKNOWN_GRACE_S"] = "2"
os.environ["POLL_INTERVAL"] = "0.2"
os.environ["HEALTH_INTERVAL"] = "3600"  # 测试中手动驱动健康监测

from app.main import app as platform_app          # noqa: E402
from app import config, service                    # noqa: E402
from node import simulator                         # noqa: E402


class RoutingTransport(httpx.AsyncBaseTransport):
    """按 host 把请求路由到对应 ASGI 应用，无需真实端口。"""

    def __init__(self, sim_a, sim_b):
        self._a = httpx.ASGITransport(app=sim_a)
        self._b = httpx.ASGITransport(app=sim_b)
        self._p = httpx.ASGITransport(app=platform_app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "testnode-a":
            return await self._a.handle_async_request(request)
        if host == "testnode-b":
            return await self._b.handle_async_request(request)
        return await self._p.handle_async_request(request)


@pytest.fixture
async def client():
    # 每台模拟器独立实例、独立内存态；每用例全新 SQLite 库
    os.makedirs(TEST_DIR, exist_ok=True)
    if os.path.exists(os.environ["PLATFORM_DB"]):
        os.remove(os.environ["PLATFORM_DB"])
    from app import db as db_mod
    db_mod._conn = None
    service._unknown_since.clear()  # 用例间隔离宽限时钟
    # ASGI 测试传输下用 0 表示直接抛连接异常；真实 uvicorn 下用长挂起复现超时
    sim_a = simulator.create_app("node-a", "模拟节点A", 8192, 0.3, 0.15,
                                 unreachable_sleep=0)
    sim_b = simulator.create_app("node-b", "模拟节点B", 8192, 0.3, 0.15,
                                 unreachable_sleep=0)
    transport = RoutingTransport(sim_a, sim_b)
    async with httpx.AsyncClient(base_url="http://platform", transport=transport,
                                 timeout=10) as c:
        # 手动初始化（不走 lifespan），并替换服务里的全局 httpx client
        service.init_db()
        await service.node_client.close()
        service.node_client._client = httpx.AsyncClient(
            base_url="http://platform", transport=transport, timeout=1.0
        )
        # 启动与生产相同的后台轮询（含健康监测，驱动失联检测与恢复自动对账）
        async def poll_loop():
            while True:
                await service.poll_loading()
                await service.poll_unverified()
                await service.poll_releasing()
                await asyncio.sleep(0.2)

        async def health_loop():
            while True:
                await service.health_monitor_once()
                await asyncio.sleep(0.5)

        poller = asyncio.create_task(poll_loop())
        health = asyncio.create_task(health_loop())
        yield c
        poller.cancel()
        health.cancel()
    await service.node_client.close()


async def wait_for(client, dep_id, statuses, timeout=12.0):
    deadline = asyncio.get_event_loop().time() + timeout
    last = None
    while asyncio.get_event_loop().time() < deadline:
        d = await client.get(f"/api/deployments/{dep_id}")
        last = d.json()
        if last["status"] in statuses:
            return last
        await asyncio.sleep(0.25)
    raise AssertionError(f"部署 {dep_id} 未进入 {statuses}，最后状态={last['status']} / {last['reason']}")


def node_free(book, nid):
    return next(n["free_mb"] for n in book if n["node_id"] == nid)


def insert_dep(dep_id, model, mem, dep_status, node_id, rep_status, epoch=1):
    """直接写库构造“崩溃前”现场：副本已预留/已下发，但加载任务随进程退出丢失。"""
    import app.db as db_mod
    ts = service.now()
    with db_mod.tx() as conn:
        conn.execute(
            "INSERT INTO deployments(id,model_name,mem_per_replica_mb,replica_count,status,"
            "reason,epoch,attempts,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (dep_id, model, mem, 1, dep_status, "已预留显存，等待加载", epoch, 1, ts, ts))
        conn.execute(
            "INSERT INTO replicas(id,deployment_id,node_id,seq,epoch,status,mem_mb,"
            "last_error,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (f"{dep_id}-r1-e{epoch}", dep_id, node_id, 1, epoch, rep_status, mem,
             None, ts, ts))


@pytest.mark.asyncio
async def test_01_full_deploy_flow_and_available(client):
    # 两台节点初始各 8192
    book = (await client.get("/api/nodes")).json()
    assert {n["node_id"] for n in book} == {"node-a", "node-b"}
    assert all(n["free_mb"] == 8192 for n in book)

    r = await client.post("/api/deployments", json={
        "model_name": "llama3-8b", "mem_per_replica_mb": 2048, "replica_count": 2})
    assert r.status_code == 201, r.text
    dep_id = r.json()["id"]

    dep = await wait_for(client, dep_id, {"AVAILABLE"})
    # 两个副本必须分处不同节点
    nodes = {rep["node_id"] for rep in dep["replicas"]}
    assert nodes == {"node-a", "node-b"}
    assert all(rep["status"] == "LOADED" for rep in dep["replicas"])

    book = (await client.get("/api/nodes")).json()
    assert all(n["used_mb"] == 2048 and n["free_mb"] == 6144 for n in book)


@pytest.mark.asyncio
async def test_02_concurrent_reservation_never_overcommits(client):
    # 每台 8192：每副本 3000，双副本部署每台占 3000；并发 3 个 -> 只有 2 个能成功
    async def submit(i):
        return await client.post("/api/deployments", json={
            "model_name": f"m{i}", "mem_per_replica_mb": 3000, "replica_count": 2})

    results = await asyncio.gather(*(submit(i) for i in range(3)))
    codes = [r.status_code for r in results]
    assert codes.count(201) == 2, codes
    assert codes.count(409) == 1, codes

    book = (await client.get("/api/nodes")).json()
    for n in book:
        assert n["used_mb"] <= n["capacity_mb"]
        assert n["used_mb"] == 6000, n  # 2 个部署 × 3000
        assert n["free_mb"] == 2192

    # 显存不足时再提交应明确拒绝
    r = await client.post("/api/deployments", json={
        "model_name": "big", "mem_per_replica_mb": 4000, "replica_count": 1})
    assert r.status_code == 409
    assert "显存不足" in r.json()["detail"]


@pytest.mark.asyncio
async def test_03_load_fail_releases_reservation_and_retry(client):
    # 让节点 A 加载失败；单副本部署落到剩余显存最多的节点（A/B 相同，排序后取 A）
    await client.post("/api/nodes/node-a/control", json={"load_fail": True})
    r = await client.post("/api/deployments", json={
        "model_name": "fail-model", "mem_per_replica_mb": 1024, "replica_count": 1})
    dep_id = r.json()["id"]
    dep = await wait_for(client, dep_id, {"FAILED"})
    rep = dep["replicas"][0]
    assert rep["status"] == "LOAD_FAILED"
    # 失败不占显存
    book = (await client.get("/api/nodes")).json()
    assert all(n["used_mb"] == 0 for n in book)

    # 清除故障后重试，应成功
    await client.post("/api/nodes/node-a/control",
                      json={"load_fail": False, "mode": "normal"})
    r2 = await client.post(f"/api/deployments/{dep_id}/retry")
    assert r2.status_code == 200
    dep2 = await wait_for(client, dep_id, {"AVAILABLE"})
    assert dep2["epoch"] == 2 and dep2["attempts"] == 2
    # 旧 epoch 副本保持终态，不复活
    old = [x for x in dep2["replicas"] if x["epoch"] == 1]
    new = [x for x in dep2["replicas"] if x["epoch"] == 2]
    assert old[0]["status"] == "LOAD_FAILED" and new[0]["status"] == "LOADED"


@pytest.mark.asyncio
async def test_04_unreachable_timeout_keeps_hold_then_reconcile_releases(client):
    await client.post("/api/nodes/node-b/control", json={"mode": "unreachable"})
    r = await client.post("/api/deployments", json={
        "model_name": "ghost", "mem_per_replica_mb": 1500, "replica_count": 2})
    dep_id = r.json()["id"]

    # A 正常加载，B 失联 -> 部署 UNVERIFIED，B 上显存保留占用
    dep = await wait_for(client, dep_id, {"UNVERIFIED"}, timeout=12)
    await service.health_monitor_once()  # 驱动一次探测周期，确认失联被健康监测捕获
    by_node = {x["node_id"]: x for x in dep["replicas"]}
    assert by_node["node-b"]["status"] == "TIMEOUT"
    # A 可能已加载或仍在加载，但绝不能被标记为失败/释放
    assert by_node["node-a"]["status"] in {"LOADING", "LOADED"}

    book = (await client.get("/api/nodes")).json()
    b = next(n for n in book if n["node_id"] == "node-b")
    assert b["used_mb"] == 1500  # 待核期间仍占用，不可被新部署用掉

    # 该显存不能被重新分配：节点 B 剩余不足时提交双副本必被拒
    r2 = await client.post("/api/deployments", json={
        "model_name": "blocked", "mem_per_replica_mb": 7000, "replica_count": 2})
    assert r2.status_code == 409

    # 节点恢复（失联期间请求从未到达模拟器，B 上实际没有该副本）
    await client.post("/api/nodes/node-b/control", json={"mode": "normal"})
    await service.health_monitor_once()  # 触发恢复自动对账
    dep = await wait_for(client, dep_id, {"AVAILABLE", "FAILED"}, timeout=12)
    # B 上实际没有副本 -> 对账释放 -> 该副本 LOAD_FAILED/RELEASED，部署 FAILED
    assert dep["status"] == "FAILED"
    by_node = {x["node_id"]: x for x in dep["replicas"]}
    assert by_node["node-b"]["status"] in {"RELEASED", "LOAD_FAILED"}
    book = (await client.get("/api/nodes")).json()
    b = next(n for n in book if n["node_id"] == "node-b")
    assert b["used_mb"] == 0  # 确认释放后显存可重新分配


@pytest.mark.asyncio
async def test_05_late_ack_must_not_revive_old_replica(client):
    # 场景：节点 A 在加载请求在途时失联，调度器保留占用转待核；
    # 之后节点其实完成了加载（迟到回执/延迟状态变化），恢复后补确认；
    # 再撤销旧加载、重试 epoch=2，旧 epoch 的迟到状态绝不能复活旧副本。

    # 1) 失联期间发起部署 -> TIMEOUT 待核，占用保留
    await client.post("/api/nodes/node-a/control", json={"mode": "unreachable"})
    await service.health_monitor_once()  # 确认健康监测捕获失联（online=False）
    r = await client.post("/api/deployments", json={
        "model_name": "late", "mem_per_replica_mb": 512, "replica_count": 1})
    dep_id = r.json()["id"]
    dep = await wait_for(client, dep_id, {"UNVERIFIED"}, timeout=10)
    rid = dep["replicas"][0]["id"]
    assert dep["replicas"][0]["status"] == "TIMEOUT"
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-a") == 8192 - 512

    # 2) 节点恢复，但它在“失联期间”其实把副本加载好了（模拟迟到回执落地）。
    #    先写入迟到状态再恢复，避免健康循环的自动对账抢先判定为无残留。
    sim_state = client._transport._a.app.state  # type: ignore[attr-defined]
    sim_state.replicas[rid] = {
        "model": "late", "mem_mb": 512, "epoch": 1, "status": "loaded",
    }
    await client.post("/api/nodes/node-a/control", json={"mode": "normal"})
    await service.health_monitor_once()  # 探测恢复边沿，触发自动对账
    # 对账补确认 -> AVAILABLE（同一 epoch 的合法补确认，不是复活）
    (await client.post(f"/api/deployments/{dep_id}/reconcile")).json()
    dep = await wait_for(client, dep_id, {"AVAILABLE"}, timeout=8)
    assert dep["replicas"][0]["status"] == "LOADED"

    # 3) 再次失联并人工撤销旧加载，恢复后对账确认释放
    await client.post("/api/nodes/node-a/control", json={"mode": "unreachable"})
    from app import models
    import app.db as db_mod
    with db_mod.tx() as conn:
        service._set_replica(conn, rid, models.REP_UNVERIFIED, error="演练：失联待核")
        service._evaluate(conn, dep_id)
    await service.request_release(rid, "演练：撤销旧加载")
    await client.post("/api/nodes/node-a/control", json={"mode": "normal"})
    deadline = asyncio.get_event_loop().time() + 15
    dep = None
    while asyncio.get_event_loop().time() < deadline:
        await client.post(f"/api/deployments/{dep_id}/reconcile")
        dep = (await client.get(f"/api/deployments/{dep_id}")).json()
        if all(x["status"] in {"RELEASED", "LOAD_FAILED"} for x in dep["replicas"]):
            break
        await asyncio.sleep(0.25)
    assert dep is not None and all(
        x["status"] in {"RELEASED", "LOAD_FAILED"} for x in dep["replicas"])

    # 4) 模拟旧 epoch 的“迟到回执”：节点侧再次出现 epoch=1 loaded
    sim_state.replicas[rid] = {
        "model": "late", "mem_mb": 512, "epoch": 1, "status": "loaded",
    }
    # 重试 -> epoch=2；调度器只认当前 epoch，旧副本终态不变
    r3 = await client.post(f"/api/deployments/{dep_id}/retry")
    assert r3.status_code == 200
    dep = await wait_for(client, dep_id, {"AVAILABLE"}, timeout=10)
    assert dep["epoch"] == 2
    old = [x for x in dep["replicas"] if x["epoch"] == 1]
    new = [x for x in dep["replicas"] if x["epoch"] == 2]
    assert all(x["status"] in {"RELEASED", "LOAD_FAILED"} for x in old)  # 不复活
    assert new[0]["status"] == "LOADED"
    # 新副本使用不同 rid，节点侧旧记录与新记录互不干扰
    assert new[0]["id"] != rid


@pytest.mark.asyncio
async def test_06_reconcile_failure_stays_unverified_with_reason(client):
    # B 失联 -> 待核；保持失联时手动对账，必须仍是待核并说明原因
    await client.post("/api/nodes/node-b/control", json={"mode": "unreachable"})
    r = await client.post("/api/deployments", json={
        "model_name": "nv", "mem_per_replica_mb": 900, "replica_count": 2})
    dep_id = r.json()["id"]
    dep = await wait_for(client, dep_id, {"UNVERIFIED"}, timeout=12)
    rec = (await client.post(f"/api/deployments/{dep_id}/reconcile")).json()
    assert rec["status"] == "UNVERIFIED"
    assert any("失联" in note or "不可达" in note for note in rec["reconcile_notes"])
    # 显存仍保留
    book = (await client.get("/api/nodes")).json()
    b = next(n for n in book if n["node_id"] == "node-b")
    assert b["used_mb"] == 900


@pytest.mark.asyncio
async def test_07_recovery_revokes_stale_load_then_confirms_release(client):
    # 正常部署成功后，节点 B 失联，调度器侧强制转待核并撤销；
    # 恢复后对账必须撤销节点上的旧加载，确认释放后才可重新分配
    r = await client.post("/api/deployments", json={
        "model_name": "svc", "mem_per_replica_mb": 2500, "replica_count": 2})
    dep_id = r.json()["id"]
    dep = await wait_for(client, dep_id, {"AVAILABLE"}, timeout=10)
    b_rep = next(x for x in dep["replicas"] if x["node_id"] == "node-b")
    rid = b_rep["id"]

    await client.post("/api/nodes/node-b/control", json={"mode": "unreachable"})
    # 直接通过服务层把 B 副本置为待核并请求释放（模拟失联期间人工处置）
    from app import models
    import app.db as db_mod
    with db_mod.tx() as conn:
        service._set_replica(conn, rid, models.REP_UNVERIFIED, error="演练：失联待核")
        service._evaluate(conn, dep_id)
    await service.request_release(rid, "演练：撤销旧加载")

    # 失联中对账不能释放
    rec = (await client.post(f"/api/deployments/{dep_id}/reconcile")).json()
    assert rec["status"] == "UNVERIFIED"
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-b") == 8192 - 2500

    # 恢复：节点上旧加载仍在（loaded），自动/手动对账应撤销并最终确认释放。
    # 自动对账与手动对账可能由不同轮次触发，终态为 RELEASED（确认释放）或
    # LOAD_FAILED（对账确认节点无残留）——两者都不再占用显存，语义等价。
    await client.post("/api/nodes/node-b/control", json={"mode": "normal"})
    await service.health_monitor_once()
    deadline = asyncio.get_event_loop().time() + 15
    while asyncio.get_event_loop().time() < deadline:
        dep = (await client.get(f"/api/deployments/{dep_id}")).json()
        b = next(x for x in dep["replicas"] if x["node_id"] == "node-b")
        if b["status"] in {"RELEASED", "LOAD_FAILED"}:
            break
        await client.post(f"/api/deployments/{dep_id}/reconcile")
        await asyncio.sleep(0.25)
    else:
        raise AssertionError("恢复后副本未在时限内确认释放")
    dep = (await client.get(f"/api/deployments/{dep_id}")).json()
    b = next(x for x in dep["replicas"] if x["node_id"] == "node-b")
    assert b["status"] in {"RELEASED", "LOAD_FAILED"}, dep
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-b") == 8192  # 关键：确认释放后显存全额可用

    # 节点侧旧记录应为 unloaded，且重新部署能拿到显存
    r2 = await client.post("/api/deployments", json={
        "model_name": "new-tenant", "mem_per_replica_mb": 7000, "replica_count": 1})
    assert r2.status_code == 201, r2.text


@pytest.mark.asyncio
async def test_08_restart_resumes_pending_and_loading(client):
    # 启动接续：写库后未下发(PENDING)与已下发回执未落库(LOADING)的副本，
    # 沿用原副本标识与轮次恢复加载；节点侧同轮重复加载不重启。
    from app import models

    # 1) 崩溃前现场一：PENDING（已预留未下发，加载任务随进程退出丢失）
    insert_dep("dep-reboot-p", "reboot-pending", 700,
               models.DEP_PENDING, "node-a", models.REP_PENDING)

    # 2) 崩溃前现场二：LOADING（已下发但回执未落库），节点侧正在加载
    await client.post("/api/nodes/node-b/control", json={"load_delay": 2})
    r = await client.post("http://testnode-b/models/reboot-loading/load",
                          params={"replica": "dep-reboot-l-r1-e1", "epoch": 1,
                                  "mem_mb": 800})
    assert r.status_code == 202
    insert_dep("dep-reboot-l", "reboot-loading", 800,
               models.DEP_PENDING, "node-b", models.REP_LOADING)

    # 3) 模拟平台重启后的接续
    service.resume_after_restart()

    dep_p = await wait_for(client, "dep-reboot-p", {"AVAILABLE"})
    dep_l = await wait_for(client, "dep-reboot-l", {"AVAILABLE"})
    # 沿用原副本标识与轮次
    assert dep_p["replicas"][0]["id"] == "dep-reboot-p-r1-e1"
    assert dep_p["replicas"][0]["epoch"] == 1
    assert dep_l["replicas"][0]["id"] == "dep-reboot-l-r1-e1"
    assert dep_l["replicas"][0]["status"] == "LOADED"
    # 节点侧同轮重复加载未重启已进行的加载
    sim_b = client._transport._b.app.state  # type: ignore[attr-defined]
    rec = sim_b.replicas["dep-reboot-l-r1-e1"]
    assert rec["attempts"] == 1 and rec["status"] == "loaded"
    # 显存账本正确
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-a") == 8192 - 700
    assert node_free(book, "node-b") == 8192 - 800


@pytest.mark.asyncio
async def test_09_duplicate_load_same_epoch_not_restarted(client):
    # 节点幂等：同轮重复加载不得重启正在进行或已经完成的加载
    await client.post("/api/nodes/node-a/control", json={"load_delay": 1.5})
    url = "http://testnode-a/models/dup-model/load"

    r1 = await client.post(url, params={"replica": "rid-dup", "epoch": 3, "mem_mb": 100})
    assert r1.status_code == 202
    # 加载进行中重复下发同轮请求：不得重启
    await asyncio.sleep(0.3)
    r2 = await client.post(url, params={"replica": "rid-dup", "epoch": 3, "mem_mb": 100})
    assert r2.status_code == 200 and r2.json()["status"] == "loading"
    state = (await client.get("http://testnode-a/state")).json()
    assert state["replicas"]["rid-dup"]["attempts"] == 1

    # 等待完成后再次重复下发：不得重启已完成的加载
    await asyncio.sleep(1.5)
    r3 = await client.post(url, params={"replica": "rid-dup", "epoch": 3, "mem_mb": 100})
    assert r3.status_code == 200 and r3.json()["status"] == "loaded"
    state = (await client.get("http://testnode-a/state")).json()
    rec = state["replicas"]["rid-dup"]
    assert rec["attempts"] == 1 and rec["status"] == "loaded"

    # 撤销后同轮重复加载：拒绝复活，防止迟到请求重新占用显存
    await client.post("http://testnode-a/models/dup-model/unload",
                      params={"replica": "rid-dup", "epoch": 3})
    await asyncio.sleep(0.5)  # 等卸载落地（unload_delay 0.15）
    r4 = await client.post(url, params={"replica": "rid-dup", "epoch": 3, "mem_mb": 100})
    assert r4.status_code == 409

    # 不同 epoch 才是新一轮：允许覆盖重新计次
    r5 = await client.post(url, params={"replica": "rid-dup", "epoch": 4, "mem_mb": 100})
    assert r5.status_code == 202
    state = (await client.get("http://testnode-a/state")).json()
    rec = state["replicas"]["rid-dup"]
    assert rec["epoch"] == 4 and rec["attempts"] == 1


@pytest.mark.asyncio
async def test_10_inflight_load_unknown_not_released(client):
    # 旧请求仍可能到达：加载落账延迟期间节点查无副本，不得释放显存
    # 落账延迟 0.6s：> 轮询间隔 0.2s，< 调用超时 1.0s
    await client.post("/api/nodes/node-a/control", json={"load_register_delay": 0.6})
    r = await client.post("/api/deployments", json={
        "model_name": "inflight", "mem_per_replica_mb": 512, "replica_count": 1})
    assert r.status_code == 201
    dep_id = r.json()["id"]

    # 落账延迟窗口内多次观察：副本必须保持加载中，显存必须保持预留
    for _ in range(2):
        await asyncio.sleep(0.2)
        dep = (await client.get(f"/api/deployments/{dep_id}")).json()
        rep = dep["replicas"][0]
        assert rep["status"] in {"PENDING", "LOADING"}, rep
        book = (await client.get("/api/nodes")).json()
        assert node_free(book, rep["node_id"]) == 8192 - 512

    # 旧请求落账后正常完成，节点侧记录与账本一致（无泄漏）
    dep = await wait_for(client, dep_id, {"AVAILABLE"})
    rep = dep["replicas"][0]
    assert rep["status"] == "LOADED"
    sim_a = client._transport._a.app.state  # type: ignore[attr-defined]
    assert sim_a.replicas[rep["id"]]["status"] == "loaded"


@pytest.mark.asyncio
async def test_11_sustained_unknown_eventually_released(client):
    # 节点持续查无副本超过宽限期：确认未占用后才释放预留
    from app import models
    insert_dep("dep-ghost", "ghost", 600,
               models.DEP_PENDING, "node-a", models.REP_LOADING)

    # 宽限期（2s）内：保持 LOADING 与预留，不得释放
    await asyncio.sleep(1.0)
    dep = (await client.get("/api/deployments/dep-ghost")).json()
    assert dep["replicas"][0]["status"] == "LOADING"
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-a") == 8192 - 600

    # 宽限期满仍查无副本：确认节点无残留后释放
    dep = await wait_for(client, "dep-ghost", {"FAILED"}, timeout=8)
    assert dep["replicas"][0]["status"] == "LOAD_FAILED"
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-a") == 8192


@pytest.mark.asyncio
async def test_12_restart_interleaving_no_revive_no_overcommit(client):
    # 重启交错：接续与在途加载交错、重复重启，不复活旧轮副本、不让新部署超额
    # 1) 一个失败重试过的部署：epoch=1 副本为终态 LOAD_FAILED
    await client.post("/api/nodes/node-a/control", json={"load_fail": True})
    r = await client.post("/api/deployments", json={
        "model_name": "flaky", "mem_per_replica_mb": 1000, "replica_count": 1})
    dep1_id = r.json()["id"]
    await wait_for(client, dep1_id, {"FAILED"})
    await client.post("/api/nodes/node-a/control",
                      json={"load_fail": False, "mode": "normal"})
    r = await client.post(f"/api/deployments/{dep1_id}/retry")
    assert r.status_code == 200
    dep1 = await wait_for(client, dep1_id, {"AVAILABLE"})
    assert dep1["epoch"] == 2

    # 2) 新部署加载在途时，模拟平台重启接续（并重复执行，验证幂等）
    await client.post("/api/nodes/node-a/control", json={"load_delay": 1.2})
    await client.post("/api/nodes/node-b/control", json={"load_delay": 1.2})
    r = await client.post("/api/deployments", json={
        "model_name": "mid-flight", "mem_per_replica_mb": 2000, "replica_count": 2})
    assert r.status_code == 201
    dep2_id = r.json()["id"]
    service.resume_after_restart()   # 接续与在途加载交错
    service.resume_after_restart()   # 重复重启：必须幂等
    dep2 = await wait_for(client, dep2_id, {"AVAILABLE"})
    assert dep2["status"] == "AVAILABLE"

    # 3) 旧轮副本未被复活
    dep1 = (await client.get(f"/api/deployments/{dep1_id}")).json()
    old = [x for x in dep1["replicas"] if x["epoch"] == 1]
    assert old and all(x["status"] == "LOAD_FAILED" for x in old)

    # 4) 节点侧每个副本只真正加载过一次（重复下发未重启加载）
    for host in ("testnode-a", "testnode-b"):
        state = (await client.get(f"http://{host}/state")).json()
        for rid, rec in state["replicas"].items():
            assert rec["attempts"] == 1, (host, rid, rec)

    # 5) 账本不超额：node-a = flaky(1000) + mid-flight(2000)，node-b = 2000
    book = (await client.get("/api/nodes")).json()
    for n in book:
        assert n["used_mb"] <= n["capacity_mb"]
    assert node_free(book, "node-a") == 8192 - 3000
    assert node_free(book, "node-b") == 8192 - 2000

    # 6) 重启接续后新部署仍受账本约束：超额必须拒绝
    r = await client.post("/api/deployments", json={
        "model_name": "too-big", "mem_per_replica_mb": 6000, "replica_count": 2})
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_13_unverified_auto_confirm_after_recovery(client):
    # 待核副本：节点失联时保留占用，恢复后由持续确认自动补确认，无需人工对账
    await client.post("/api/nodes/node-a/control", json={"mode": "unreachable"})
    r = await client.post("/api/deployments", json={
        "model_name": "auto-confirm", "mem_per_replica_mb": 512, "replica_count": 1})
    dep_id = r.json()["id"]
    dep = await wait_for(client, dep_id, {"UNVERIFIED"}, timeout=10)
    rid = dep["replicas"][0]["id"]
    assert dep["replicas"][0]["status"] == "TIMEOUT"
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-a") == 8192 - 512  # 失联期间保留占用

    # 节点恢复，且失联期间加载其实已完成（迟到落地）
    sim_a = client._transport._a.app.state  # type: ignore[attr-defined]
    sim_a.replicas[rid] = {"model": "auto-confirm", "mem_mb": 512, "epoch": 1,
                           "status": "loaded", "attempts": 1}
    await client.post("/api/nodes/node-a/control", json={"mode": "normal"})

    # 不手动对账：持续确认自动补确认
    dep = await wait_for(client, dep_id, {"AVAILABLE"}, timeout=10)
    assert dep["replicas"][0]["status"] == "LOADED"
