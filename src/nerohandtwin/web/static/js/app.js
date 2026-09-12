// NERO 手势跟随控制台前端
// 三库分工：P5=相机叠加层（骨架/十字/指距线）、Three.js=3D 数字孪生、D3=遥测仪表
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const HAND_EDGES = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],[0,17],
];

// 叠加元素开关（设置面板控制，localStorage 持久化）
const settings = Object.assign({
  showSkeleton: true,   // 手部骨架
  showCross: true,      // 中心十字
  showPalm: true,       // 掌心标记
  showFollowLine: true, // 跟随连线（掌心↔十字）
  showGapLine: true,    // 指距线（拇指-中指）
  light: false,         // 亮色主题
}, JSON.parse(localStorage.getItem("nero-ui") || "{}"));

function saveSettings() {
  localStorage.setItem("nero-ui", JSON.stringify(settings));
}

let latest = null;      // 最新 WS 状态帧
let ws = null;
let wsOk = false;

// ---------- WebSocket ----------
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => {
    wsOk = true;
    document.getElementById("ws-state").textContent = "已连接";
    document.getElementById("ws-state").classList.add("on");
  };
  ws.onclose = () => {
    wsOk = false;
    document.getElementById("ws-state").textContent = "断线重连中…";
    document.getElementById("ws-state").classList.remove("on");
    setTimeout(connectWS, 1500);
  };
  ws.onmessage = (ev) => {
    latest = JSON.parse(ev.data);
    updateHUD(latest);
    if (trend) trend.update(latest);
    syncArmParams(latest);
  };
}

function sendCmd(cmd, extra = {}) {
  if (ws && wsOk) ws.send(JSON.stringify(Object.assign({ cmd }, extra)));
}

// ---------- HUD（顶栏/文本） ----------
function updateHUD(d) {
  const badge = document.getElementById("mode-badge");
  const rec = document.getElementById("rec-badge");
  badge.textContent = d.mode;
  badge.className = "badge";
  if (d.recording) { badge.classList.add("badge-rec"); rec.classList.remove("hidden"); }
  else {
    badge.classList.add(d.in_follow ? "badge-follow" : "badge-idle");
    rec.classList.add("hidden");
  }
  document.getElementById("fps").textContent = `${d.fps} FPS`;
  document.getElementById("hud-gesture").textContent =
    d.gesture_stable + (d.gesture_pending ? ` →${d.gesture_pending} ${Math.round(d.gesture_pending_progress*100)}%` : "");
  document.getElementById("hud-gap").textContent =
    d.hand.gap_m !== null ? Math.round(d.hand.gap_m * 1000) : "--";
  document.getElementById("hud-gripper").textContent = d.gripper_mm;

  const lines = [];
  const tcp = d.arm && d.arm.tcp;
  lines.push(tcp
    ? `TCP=[${tcp.map(x => x.toFixed(2)).join(", ")}]m  夹爪=${Math.round(d.gripper_mm)}mm`
    : `TCP=--  夹爪=${Math.round(d.gripper_mm)}mm`);
  if (d.in_follow && d.gimbal_err) {
    const [eu, ev, z] = d.gimbal_err;
    lines.push(`伺服误差: eu=${eu.toFixed(2)} ev=${ev.toFixed(2)} 深度=${z.toFixed(2)}m`);
  }
  document.getElementById("status-lines").textContent = lines.join("\n");
}

// ---------- P5：相机叠加层 ----------
// 画布尺寸跟随视频 object-fit:contain 的实际显示区域（含黑边补偿），
// 保证骨架点与画面像素严格对齐。
const overlaySketch = (p) => {
  const wrap = document.querySelector(".cam-wrap");
  p.setup = () => {
    p.pixelDensity(1);
    p.createCanvas(640, 480).parent(wrap);
    p.noLoop();
  };
  const fitRect = () => {
    const [fh, fw] = latest ? latest.frame_shape : [480, 640];
    const s = Math.min(wrap.clientWidth / fw, wrap.clientHeight / fh);
    return { w: Math.round(fw * s), h: Math.round(fh * s) };
  };
  const drawFrame = () => {
    p.clear();
    if (!latest || !latest.hand.present || !latest.hand.landmarks) return;

    const [fh, fw] = latest.frame_shape;
    const sx = p.width / fw, sy = p.height / fh;
    const lm = latest.hand.landmarks.map(([u, v]) => [u * sx, v * sy]);
    const palm = lm[9], tip4 = lm[4], tip12 = lm[12];
    const light = settings.light;

    if (settings.showSkeleton) {
      p.stroke(light ? 60 : 200, light ? 66 : 205, light ? 74 : 214, 200);
      p.strokeWeight(1.5);
      for (const [a, b] of HAND_EDGES) p.line(lm[a][0], lm[a][1], lm[b][0], lm[b][1]);
      p.noStroke();
      for (const [x, y] of lm) { p.fill(light ? 90 : 144, light ? 96 : 178, light ? 102 : 233); p.circle(x, y, 6); }
    }

    if (settings.showPalm) {
      p.fill(230, 57, 70); p.circle(palm[0], palm[1], 12);
    }

    if (settings.showGapLine) {
      p.stroke(230, 57, 70); p.strokeWeight(2);
      p.line(tip4[0], tip4[1], tip12[0], tip12[1]);
      p.noStroke();
      if (latest.hand.gap_m !== null) {
        p.fill(230, 57, 70);
        p.text(`${Math.round(latest.hand.gap_m * 1000)}mm`,
               (tip4[0]+tip12[0])/2 + 8, (tip4[1]+tip12[1])/2);
      }
    }

    if (latest.in_follow && settings.showCross) {
      const cx = p.width/2, cy = p.height/2;
      p.stroke(255, 255, 0); p.strokeWeight(2);
      p.line(cx-16, cy, cx+16, cy); p.line(cx, cy-16, cx, cy+16);
      p.noFill(); p.circle(cx, cy, 44);
    }
    if (latest.in_follow && settings.showFollowLine) {
      p.stroke(255, 255, 255, 180); p.strokeWeight(1);
      p.line(palm[0], palm[1], p.width/2, p.height/2);
    }
  };
  p.draw = () => {
    const { w, h } = fitRect();
    if (w > 0 && h > 0 && (w !== p.width || h !== p.height)) p.resizeCanvas(w, h);
    if (p.canvas) {
      p.canvas.style.left = `${Math.round((wrap.clientWidth - p.width) / 2)}px`;
      p.canvas.style.top = `${Math.round((wrap.clientHeight - p.height) / 2)}px`;
    }
    drawFrame();
  };
  // 每帧驱动（不用 requestAnimationFrame 以配合 WS 帧率）
  setInterval(() => p.redraw(), 33);
};
new p5(overlaySketch);

// ---------- Three.js：3D 数字孪生 ----------
const MDH = [
  [0.138, 0, 0, 0],
  [0, 0, Math.PI/2, Math.PI],
  [0.31, 0, Math.PI/2, Math.PI],
  [0, 0, Math.PI/2, Math.PI],
  [0.27001, 0, Math.PI/2, Math.PI/2],
  [0, 0, Math.PI/2, Math.PI/2],
  [0.0235, 0, Math.PI/2, 0],
];

// 正运动学：返回 7 个关节坐标系的 4x4 齐次矩阵（位置 + 姿态）
function neroFK(q) {
  const frames = [];
  let T = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  const mul = (A, B) => {
    const C = Array.from({length: 4}, () => [0,0,0,0]);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++)
      for (let k = 0; k < 4; k++) C[i][j] += A[i][k]*B[k][j];
    return C;
  };
  const link = (d, a, al, th) => {
    const [ca, sa, ct, st] = [Math.cos(al), Math.sin(al), Math.cos(th), Math.sin(th)];
    return [[ct,-st,0,a],[ca*st,ca*ct,-sa,-sa*d],[sa*st,sa*ct,ca,ca*d],[0,0,0,1]];
  };
  for (let i = 0; i < 7; i++) {
    const [d, a, al, off] = MDH[i];
    T = mul(T, link(d, a, al, q[i] + off));
    frames.push(T);
  }
  return frames;
}

let three = null;
function initThree() {
  const wrap = document.getElementById("three-wrap");
  const w = wrap.clientWidth, h = wrap.clientHeight;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(settings.light ? 0xe9ecf2 : 0x0b0d10);

  // FK/CAD 均为 Z-up，Three.js 为 Y-up：机械臂整体放进 rig 容器做旋转
  const rig = new THREE.Group();
  rig.rotation.x = -Math.PI / 2;
  scene.add(rig);

  const camera = new THREE.PerspectiveCamera(46, w / h, 0.05, 12);
  camera.position.set(0.95, 0.8, 1.15);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  wrap.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.42, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.4;
  controls.maxDistance = 4.0;
  controls.maxPolarAngle = Math.PI * 0.495;  // 视角不钻到地面以下
  controls.update();

  // 灯光：半球环境光 + 主光（投影）+ 红色轮廓光
  const hemi = new THREE.HemisphereLight(0xffffff, 0x30363f, settings.light ? 1.25 : 1.5);
  scene.add(hemi);
  const key = new THREE.DirectionalLight(0xffffff, 2.0);
  key.position.set(1.2, 1.8, 1.0);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.camera.left = -0.8; key.shadow.camera.right = 0.8;
  key.shadow.camera.top = 0.8; key.shadow.camera.bottom = -0.8;
  key.shadow.camera.far = 5;
  key.shadow.bias = -0.001;
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xe63946, 1.3);
  rim.position.set(-1.2, 0.6, -0.9);
  scene.add(rim);
  const fill = new THREE.DirectionalLight(0x8fb4ff, 0.5);
  fill.position.set(-0.8, 0.9, 1.2);
  scene.add(fill);

  // 地面：明暗两套网格（主题切换显隐）+ 阴影接收面
  const gridDark = new THREE.GridHelper(2.0, 20, 0x3a4150, 0x232830);
  const gridLight = new THREE.GridHelper(2.0, 20, 0x9aa2b0, 0xc6ccd8);
  gridLight.visible = settings.light;
  gridDark.visible = !settings.light;
  scene.add(gridDark, gridLight);
  const shadowPlane = new THREE.Mesh(
    new THREE.PlaneGeometry(3, 3),
    new THREE.ShadowMaterial({ opacity: settings.light ? 0.18 : 0.32 })
  );
  shadowPlane.rotation.x = -Math.PI / 2;
  shadowPlane.receiveShadow = true;
  scene.add(shadowPlane);

  const matLinkDark = new THREE.MeshStandardMaterial({ color: 0x3a4049, metalness: 0.45, roughness: 0.5 });
  const matLinkSilver = new THREE.MeshStandardMaterial({ color: 0xd6dae2, metalness: 0.55, roughness: 0.35 });
  const matJoint = new THREE.MeshStandardMaterial({ color: 0x2b3038, metalness: 0.45, roughness: 0.45 });
  const matRed = new THREE.MeshStandardMaterial({
    color: 0xe63946, metalness: 0.4, roughness: 0.35,
    emissive: 0xe63946, emissiveIntensity: 0,   // 跟随中呼吸发光
  });

  const cyl = (r1, r2, len, mat) =>
    new THREE.Mesh(new THREE.CylinderGeometry(r1, r2, len, 24), mat);
  const linkMeshes = [
    cyl(0.055, 0.045, 0.138, matLinkDark),
    cyl(0.045, 0.038, 0.31, matLinkSilver),
    cyl(0.038, 0.032, 0.270, matLinkDark),
    cyl(0.032, 0.024, 0.0235, matLinkSilver),
  ];
  const jointMeshes = [];
  for (let i = 0; i < 8; i++) {
    const r = [0.07, 0.055, 0.042, 0.042, 0.036, 0.032, 0.028, 0.024][i];
    jointMeshes.push(new THREE.Mesh(new THREE.SphereGeometry(r, 24, 18), matJoint));
  }
  const finger1 = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.09, 0.02), matRed);
  const finger2 = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.09, 0.02), matRed);
  const allMeshes = [...linkMeshes, ...jointMeshes, finger1, finger2];
  for (const m of allMeshes) m.castShadow = true;
  rig.add(...allMeshes);

  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.11, 0.03, 32), matLinkDark);
  base.position.y = 0.015;
  base.castShadow = true;
  scene.add(base);
  // 底座红色装饰环（跟随状态指示灯）
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.1, 0.006, 12, 48),
    new THREE.MeshStandardMaterial({ color: 0xe63946, emissive: 0xe63946, emissiveIntensity: 0.25 })
  );
  ring.rotation.x = Math.PI / 2;
  ring.position.y = 0.032;
  scene.add(ring);

  const up = new THREE.Vector3(0, 1, 0);
  function setLink(mesh, a, b) {
    const va = new THREE.Vector3(...a), vb = new THREE.Vector3(...b);
    mesh.position.copy(va.clone().add(vb).multiplyScalar(0.5));
    const dir = vb.clone().sub(va);
    const len = Math.max(dir.length(), 1e-4);
    mesh.scale.y = len / mesh.geometry.parameters.height;
    mesh.quaternion.setFromUnitVectors(up, dir.normalize());
  }
  const pos = (T) => [T[0][3], T[1][3], T[2][3]];
  const _m4 = new THREE.Matrix4();
  const flangeQuat = (T) => {
    _m4.set(T[0][0],T[0][1],T[0][2],T[0][3],
            T[1][0],T[1][1],T[1][2],T[1][3],
            T[2][0],T[2][1],T[2][2],T[2][3],
            0,0,0,1);
    return new THREE.Quaternion().setFromRotationMatrix(_m4);
  };
  // 官方模型（scripts/convert_urdf_model.py 产物）：URDF 关节树 + 逐 link STL
  const MODEL_VERSION = "2026-09-11-7";  // 模型资产变更时递增，防止浏览器缓存旧 STL
  const official = {
    ready: false, rev: Array(7).fill(null), finger1: null, finger2: null,
  };

  function poseArm(q) {
    const frames = neroFK(q);
    const f = frames.map(pos);
    jointMeshes.forEach((m, i) => m.position.set(...f[Math.min(i, 6)]));
    setLink(linkMeshes[0], [0, 0, 0], f[1]);
    setLink(linkMeshes[1], f[1], f[2]);
    setLink(linkMeshes[2], f[3], f[4]);
    setLink(linkMeshes[3], f[5], f[6]);
    // 夹爪两指：沿法兰坐标系开合 + 随腕部旋转
    const quat = flangeQuat(frames[6]);
    const fp = new THREE.Vector3(...f[6]);
    const fwd = new THREE.Vector3().subVectors(
      new THREE.Vector3(...f[6]), new THREE.Vector3(...f[5])).normalize();
    const openAxis = new THREE.Vector3(0, 1, 0).applyQuaternion(quat);
    const opening = 0.005 + (latest ? latest.gripper_opening : 1.0) * 0.045;
    const tipBase = fp.addScaledVector(fwd, 0.1);
    finger1.position.copy(tipBase).addScaledVector(openAxis, opening);
    finger2.position.copy(tipBase).addScaledVector(openAxis, -opening);
    finger1.quaternion.copy(quat);
    finger2.quaternion.copy(quat);
    // 官方 URDF 模型：revolute 关节按 SDK 关节角（与 MuJoCo 同名直映射）、
    // prismatic 夹爪指按开度（行程 0.1m = 每指 0.05m）
    if (official.ready) {
      for (let i = 0; i < 7; i++) {
        const j = official.rev[i];
        if (!j) continue;
        j.quaternion.copy(j.userData.originQuat)
          .multiply(_qTmp.setFromAxisAngle(j.userData.axis, curQ[i]));
      }
      const open = (latest ? latest.gripper_opening : 1.0) * 0.05;
      for (const f of [official.finger1, official.finger2]) {
        if (f) f.position.copy(f.userData.originPos)
          .addScaledVector(f.userData.parentAxis, open * f.userData.qSign);
      }
    }
  }

  // 官方模型加载：URDF 关节树（与真机/MuJoCo 同一运动学），失败回退程序化臂
  const LINK_MATS = {
    base_link: matLinkDark, link1: matLinkDark, link2: matLinkSilver,
    link3: matLinkDark, link4: matLinkSilver, link5: matLinkDark,
    link6: matLinkSilver, link7: matLinkDark,
    gripper_flange: matLinkDark, gripper_base: matLinkDark,
    gripper_link1: matRed, gripper_link2: matRed,
  };
  const _qTmp = new THREE.Quaternion();
  const _eTmp = new THREE.Euler();
  async function loadOfficialModel() {
    const resp = await fetch(`/static/model/urdf.json?v=${MODEL_VERSION}`);
    if (!resp.ok) throw new Error(`urdf.json HTTP ${resp.status}`);
    const m = await resp.json();
    const loader = new STLLoader();
    const linkObjs = {};
    for (const [name, info] of Object.entries(m.links)) {
      const geo = await loader.loadAsync(`/static/model/${info.mesh}?v=${MODEL_VERSION}`);
      const mesh = new THREE.Mesh(geo, LINK_MATS[name] || matLinkSilver);
      mesh.castShadow = true;
      const holder = new THREE.Group();
      holder.position.set(...info.mesh_origin.xyz);
      const [r, p, y] = info.mesh_origin.rpy;
      holder.quaternion.setFromEuler(_eTmp.set(r, p, y, "ZYX"));  // URDF rpy = Rz·Ry·Rx
      holder.add(mesh);
      linkObjs[name] = holder;
    }
    const revByName = {}, priByName = {};
    for (const j of m.joints) {
      const jg = new THREE.Group();
      jg.position.set(...j.origin.xyz);
      const [r, p, y] = j.origin.rpy;
      jg.quaternion.setFromEuler(_eTmp.set(r, p, y, "ZYX"));
      jg.userData.originQuat = jg.quaternion.clone();
      jg.userData.originPos = jg.position.clone();
      jg.userData.axis = new THREE.Vector3(...j.axis).normalize();
      // 棱柱位移必须先经关节 origin rpy 旋转到父系方向（URDF 语义：
      // child = origin_translation + origin_rotation·(axis·q)）。直接用原始
      // axis 会让指板沿夹爪长度轴滑动而非开合轴 → 指板位置错误。
      jg.userData.parentAxis = jg.userData.axis.clone()
        .applyQuaternion(jg.userData.originQuat);
      // 开合符号：两指限位一侧正一侧负（joint1 [0,0.05] / joint2 [-0.05,0]），
      // 同一 open 值下反向平移才能张开。
      jg.userData.qSign = (j.limit && j.limit[1] > 0) ? 1 : -1;
      if (j.type === "revolute") revByName[j.name] = jg;
      else if (j.type === "prismatic") priByName[j.name] = jg;
      const parent = j.parent === "world" ? rig : (linkObjs[j.parent] || rig);
      parent.add(jg);
      if (linkObjs[j.child]) jg.add(linkObjs[j.child]);
    }
    for (let i = 0; i < 7; i++) official.rev[i] = revByName[`joint${i + 1}`] || null;
    official.finger1 = priByName["gripper_joint1"] || null;
    official.finger2 = priByName["gripper_joint2"] || null;
    official.ready = !!official.rev[0];
    if (official.ready) {
      for (const msh of [...linkMeshes, ...jointMeshes, finger1, finger2, base, ring]) msh.visible = false;
      controls.target.set(0, 0.34, 0);
      camera.position.set(0.62, 0.5, 0.72);
      controls.update();
      // 面板副标题显示模型版本：用户一眼确认浏览器加载的是哪份模型资产
      const sub = document.querySelector(".panel-3d .panel-sub");
      if (sub) sub.textContent = `模型 ${MODEL_VERSION} · 左键旋转 / 右键平移 / 滚轮缩放`;
      window.__nero3d = { scene, rig, official, THREE };   // 几何诊断句柄
      console.info(`[3D] 官方 URDF 模型已加载（关节树驱动）v=${MODEL_VERSION}`);
    }
  }
  loadOfficialModel().catch((err) =>
    console.warn("[3D] 官方模型加载失败，保留程序化臂:", err));

  // 平滑插值：WS 每 100ms 给目标角，RAF 每帧指数趋近，运动丝滑无跳变
  const readyQ = [0, -0.6109, 0, 2.0071, 0, 0, 0.2618];
  let targetQ = readyQ.slice();
  let curQ = readyQ.slice();
  let lastT = performance.now();
  function frame(t) {
    const dt = Math.min(0.05, (t - lastT) / 1000);
    lastT = t;
    const k = 1 - Math.exp(-dt * 10);
    for (let i = 0; i < 7; i++) curQ[i] += (targetQ[i] - curQ[i]) * k;
    poseArm(curQ);
    const following = !!(latest && latest.in_follow);
    matRed.emissiveIntensity = following ? 0.45 + 0.3 * Math.sin(t * 0.006) : 0;
    ring.material.emissiveIntensity = following ? 0.9 + 0.5 * Math.sin(t * 0.006) : 0.25;
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  poseArm(curQ);
  requestAnimationFrame(frame);

  three = {
    update(q) {   // WS 推送的目标关节角（仅记录目标，插值由 RAF 完成）
      if (q && q.length >= 7) targetQ = q.slice(0, 7);
    },
    setTheme(light) {
      scene.background = new THREE.Color(light ? 0xe9ecf2 : 0x0b0d10);
      gridLight.visible = light;
      gridDark.visible = !light;
      shadowPlane.material.opacity = light ? 0.18 : 0.32;
      hemi.intensity = light ? 1.25 : 1.5;
    },
    onResize() {
      const nw = wrap.clientWidth, nh = wrap.clientHeight;
      if (nw === 0 || nh === 0) return;
      camera.aspect = nw / nh;
      camera.updateProjectionMatrix();
      renderer.setSize(nw, nh);
    },
  };
  window.addEventListener("resize", () => three.onResize());
}

// ---------- D3 仪表 ----------
function initGauges() {
  const pal = () => settings.light
    ? { track: "#dfe3ea", fg: "#3f4756", text: "#d7263d", label: "#66707e", hot: "#e63946" }
    : { track: "#3a4150", fg: "#c8cdd6", text: "#e63946", label: "#9aa2b0", hot: "#e63946" };
  const setters = [];
  const mkGauge = (sel, label, min, max, fmt) => {
    const svg = d3.select(sel).attr("viewBox", "0 0 120 84");
    const arc = d3.arc().innerRadius(34).outerRadius(48)
      .startAngle(-Math.PI / 2).cornerRadius(3);
    const cx = 60, cy = 66;
    const c = pal();
    const track = svg.append("path")
      .attr("transform", `translate(${cx},${cy})`)
      .attr("fill", "none")
      .attr("stroke", c.track)
      .attr("stroke-width", 11)
      .attr("stroke-linecap", "round")
      .attr("d", arc({ endAngle: Math.PI / 2 }));
    const fg = svg.append("path")
      .attr("transform", `translate(${cx},${cy})`)
      .attr("fill", "none")
      .attr("stroke", c.fg)
      .attr("stroke-width", 9)
      .attr("stroke-linecap", "round");
    const txt = svg.append("text").attr("x", cx).attr("y", cy - 2)
      .attr("text-anchor", "middle").attr("fill", c.text)
      .attr("font-size", 17).attr("font-weight", 700);
    const lbl = svg.append("text").attr("x", cx).attr("y", cy + 14)
      .attr("text-anchor", "middle").attr("fill", c.label).attr("font-size", 10);
    lbl.text(label);
    let lastV = min;
    const set = (v) => {
      lastV = v;
      const cc = pal();
      const frac = Math.max(0, Math.min(1, (v - min) / (max - min)));
      fg.attr("d", arc({ endAngle: -Math.PI / 2 + frac * Math.PI }))
        .attr("stroke", frac > 0.92 ? cc.hot : cc.fg);
      txt.text(fmt(v));
    };
    set._recolor = () => {
      const cc = pal();
      track.attr("stroke", cc.track);
      txt.attr("fill", cc.text);
      lbl.attr("fill", cc.label);
      set(lastV);   // 用主题色重绘当前值
    };
    setters.push(set);
    return set;
  };
  const gauges = {
    gripper: mkGauge("#gauge-gripper", "开度/100mm", 0, 0.1, (v) => `${Math.round(v*1000)}`),
    j1: mkGauge("#gauge-j1", "J1 底座", -1.2, 1.2, (v) => v.toFixed(2)),
    j2: mkGauge("#gauge-j2", "J2 肩", -1.6, 0.4, (v) => v.toFixed(2)),
    j4: mkGauge("#gauge-j4", "J4 肘", -0.95, 2.1, (v) => v.toFixed(2)),
  };
  gauges.setTheme = () => setters.forEach((s) => s._recolor());
  return gauges;
}

// ---------- D3 伺服误差趋势线（eu/ev/深度 滚动波形） ----------
function initTrend() {
  const el = document.getElementById("servo-trend");
  if (!el) return null;
  const W = 320, H = 56, N = 150;
  const svg = d3.select(el).append("svg")
    .attr("width", "100%").attr("height", "100%")
    .attr("viewBox", `0 0 ${W} ${H}`)
    .attr("preserveAspectRatio", "none");
  const series = [[], [], []];   // eu / ev / z_err
  const mid = svg.append("line");
  const paths = [0, 1, 2].map(() => svg.append("path").attr("fill", "none").attr("stroke-width", 1.5));
  const colors = () => settings.light
    ? ["#e63946", "#2a6fd6", "#d97706"]
    : ["#e63946", "#6ea8fe", "#fbbf24"];
  const lineGen = d3.line().x((_, i) => i * (W / (N - 1)));
  const scaleV = [H * 0.9, H * 0.9, H * 1.4];   // eu/ev ±0.5 满幅；z_err ±0.3m 满幅
  return {
    update(d) {
      if (!d || !d.gimbal_err) return;
      for (let i = 0; i < 3; i++) {
        series[i].push(d.gimbal_err[i]);
        if (series[i].length > N) series[i].shift();
      }
      const cc = colors();
      const axis = settings.light ? "#c0c6d2" : "#3a4150";
      mid.attr("x1", 0).attr("x2", W).attr("y1", H / 2).attr("y2", H / 2)
        .attr("stroke", axis).attr("stroke-width", 1).attr("stroke-dasharray", "3,3");
      for (let i = 0; i < 3; i++) {
        paths[i].attr("stroke", cc[i])
          .attr("d", lineGen.y((v) => H / 2 - v * scaleV[i])(series[i]));
      }
    },
  };
}

// ---------- 设置面板 ----------
function applyTheme() {
  document.body.classList.toggle("light", settings.light);
  const btn = document.getElementById("btn-theme");
  if (btn) btn.textContent = settings.light ? "🌙 暗色" : "☀️ 亮色";
  if (three) three.setTheme(settings.light);
  if (gauges) gauges.setTheme();
}

function applyOverlaySettings() {
  // P5 sketch 每帧直接读 settings 对象，无需额外处理
}

function buildSettingsPanel() {
  const panel = document.getElementById("settings-panel");
  const secTitle = (t) => {
    const d = document.createElement("div");
    d.className = "settings-section-title";
    d.textContent = t;
    panel.appendChild(d);
  };
  secTitle("显示叠加层");
  const items = [
    ["showSkeleton", "手部骨架"],
    ["showCross", "中心十字"],
    ["showPalm", "掌心标记"],
    ["showFollowLine", "跟随连线"],
    ["showGapLine", "指距线"],
  ];
  for (const [key, label] of items) {
    const row = document.createElement("label");
    row.className = "setting-row";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = settings[key];
    cb.onchange = () => { settings[key] = cb.checked; saveSettings(); applyOverlaySettings(); };
    row.appendChild(cb);
    row.appendChild(document.createTextNode(label));
    panel.appendChild(row);
  }
  const themeBtn = document.createElement("button");
  themeBtn.className = "btn";
  themeBtn.id = "btn-theme";
  themeBtn.textContent = settings.light ? "🌙 暗色" : "☀️ 亮色";
  themeBtn.onclick = () => {
    settings.light = !settings.light;
    saveSettings();
    applyTheme();
  };
  panel.appendChild(themeBtn);
}

// ---------- 机械臂参数（运行时下发，主循环内生效） ----------
const ARM_PARAMS = [
  // [key, 标签, min, max, step, 单位, 初始值(未收到服务端参数前)]
  ["speed_percent", "全局速度", 1, 100, 5, "%", 10],
  ["max_joint_rate", "跟随关节限速", 0.1, 1.5, 0.05, "rad/s", 0.5],
  ["k_yaw", "方位增益", 0.1, 2.0, 0.05, "", 0.9],
  ["k_pitch", "俯仰增益", 0.1, 2.0, 0.05, "", 0.7],
  ["k_depth", "深度增益", 0.1, 2.0, 0.05, "", 0.6],
  ["dz_u", "方位死区", 0.0, 0.15, 0.01, "", 0.04],
  ["dz_v", "俯仰死区", 0.0, 0.15, 0.01, "", 0.05],
  ["dz_z", "深度死区", 0.0, 0.15, 0.01, "m", 0.04],
  ["z_ref", "手-相机参考距离", 0.25, 0.80, 0.01, "m", 0.40],
];
const armParamInputs = {};   // key -> input 元素
let armParamsSynced = false; // 首次收到服务端参数后回填滑杆

function buildArmParams() {
  const panel = document.getElementById("settings-panel");
  const sec = document.createElement("div");
  sec.className = "settings-section-title";
  sec.textContent = "机械臂参数（跟随速度）";
  panel.appendChild(sec);
  for (const [key, label, min, max, step, unit, init] of ARM_PARAMS) {
    const row = document.createElement("div");
    row.className = "param-row";
    const head = document.createElement("div");
    head.className = "param-head";
    head.innerHTML = `<span>${label}</span><b data-v="${key}">${init}${unit}</b>`;
    const inp = document.createElement("input");
    inp.type = "range";
    inp.min = min; inp.max = max; inp.step = step; inp.value = init;
    inp.oninput = () => {
      head.querySelector("b").textContent = `${parseFloat(inp.value)}${unit}`;
    };
    row.appendChild(head);
    row.appendChild(inp);
    panel.appendChild(row);
    armParamInputs[key] = { inp, head };
  }
  const hint = document.createElement("div");
  hint.className = "param-hint";
  hint.textContent = "提速三层（按序逐级上调）：① 全局速度% 是 SDK 总倍率（默认 10 为联调保守值，建议 30→50）；② 关节限速限制每秒角增量（0.5≈29°/s，建议 0.8→1.2）；③ 增益越大响应越快、越易抖。死区越小越灵敏。";
  panel.appendChild(hint);
  const applyBtn = document.createElement("button");
  applyBtn.className = "btn btn-primary";
  applyBtn.id = "btn-apply-params";
  applyBtn.textContent = "应用机械臂参数";
  applyBtn.onclick = () => {
    const params = {};
    for (const [key] of ARM_PARAMS) params[key] = parseFloat(armParamInputs[key].inp.value);
    sendCmd("set_params", { params });
    applyBtn.textContent = "已下发 ✓";
    setTimeout(() => { applyBtn.textContent = "应用机械臂参数"; }, 1200);
  };
  panel.appendChild(applyBtn);
}

function syncArmParams(d) {
  // 服务端当前值回填（只做一次，避免覆盖用户拖动中的滑杆）
  if (!d.params || armParamsSynced) return;
  const g = d.params.gimbal || {};
  const vals = { speed_percent: d.params.speed_percent };
  for (const [key] of ARM_PARAMS) {
    if (key !== "speed_percent" && g[key] !== undefined) vals[key] = g[key];
  }
  for (const [key] of ARM_PARAMS) {
    const v = vals[key];
    const el = armParamInputs[key];
    if (el && v !== null && v !== undefined) {
      el.inp.value = v;
      el.head.querySelector("b").textContent =
        `${parseFloat(v)}${ARM_PARAMS.find(x => x[0] === key)[5]}`;
    }
  }
  armParamsSynced = true;
}

// ---------- 控制按钮 ----------
function wireButtons() {
  document.getElementById("btn-follow").onclick = () => sendCmd("follow_toggle");
  document.getElementById("btn-reset").onclick = () => sendCmd("reset");
  document.getElementById("btn-anchor").onclick = () => sendCmd("anchor");
  document.getElementById("btn-record").onclick = () => sendCmd("record");
  document.getElementById("btn-play").onclick = () => sendCmd("playback");
  document.getElementById("btn-shutdown").onclick = () => sendCmd("shutdown");
}

// ---------- 启动 ----------
// 任一子系统初始化失败不拖垮其余模块（错误显示在页面角落）
function bootError(tag, err) {
  console.error(`[${tag}]`, err);
  const el = document.createElement("div");
  el.className = "boot-error";
  el.textContent = `${tag} 初始化失败: ${err && err.message ? err.message : err}`;
  document.body.appendChild(el);
}

let gauges = null;
let trend = null;
connectWS();
try { initThree(); } catch (err) { bootError("3D孪生", err); }
try { gauges = initGauges(); } catch (err) { bootError("仪表", err); }
try { trend = initTrend(); } catch (err) { bootError("趋势线", err); }
wireButtons();
buildSettingsPanel();
buildArmParams();
applyTheme();

setInterval(() => {
  if (!latest) return;
  try {
    if (gauges) {
      gauges.gripper(latest.gripper_opening * 0.1);
      if (latest.arm.joint_angles) {
        gauges.j1(latest.arm.joint_angles[0]);
        gauges.j2(latest.arm.joint_angles[1]);
        gauges.j4(latest.arm.joint_angles[3]);
      }
    }
    if (three && latest.arm.joint_angles) three.update(latest.arm.joint_angles);
  } catch (err) { /* 单帧渲染失败不影响下一帧 */ }
}, 100);
