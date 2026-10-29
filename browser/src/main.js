import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {toCreasedNormals} from 'three/addons/utils/BufferGeometryUtils.js';
import './style.css';
const $=name=>document.getElementById(name);
const canvas=$('viewport');
const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(42,1,0.02,200);
camera.up.set(0,0,1);camera.position.set(3.25,0,1.47);
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
renderer.outputColorSpace=THREE.SRGBColorSpace;
canvas.appendChild(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);
controls.target.set(0,0,.75);controls.enableDamping=true;controls.update();
// Camera modes: world-front, torso yaw tracking with a level horizon,
// and unrestricted free orbit. Both tracking modes use world +Z for up.
const cameraModes=['follow','torso','free'];
let cameraMode='follow';
const rootPosition=new THREE.Vector3(0,0,.8);
const followTarget=new THREE.Vector3(0,0,.75);
const targetGoal=new THREE.Vector3();
const eyeGoal=new THREE.Vector3();
const frontalOffset=new THREE.Vector3(3.25,0,.72);
const torsoPosition=new THREE.Vector3(0,0,1.1);
const torsoNormal=new THREE.Vector3(1,0,0);
const torsoHeading=new THREE.Vector3(1,0,0);
const worldUp=new THREE.Vector3(0,0,1);
const cameraForward=new THREE.Vector3();
const horizontalLookAt=new THREE.Vector3();
const cameraDistance=3.25;
const TORSO_HEADING_MIN=0.15;  // freeze yaw when chest normal points nearly vertical
const cameraButton=$('camera-mode');
const cameraLabels={
  follow:'Камера: спереди',
  torso:'Камера: по торсу',
  free:'Камера: свободная'
};
const cameraDescriptions={
  follow:'Мировой фронтальный ракурс; слежение за перемещением',
  torso:'Камера горизонтальна, перед торсом, поворачивается только по курсу корпуса',
  free:'Вращение, зум и перемещение камеры мышью'
};
const updateCameraButton=()=>{
  cameraButton.dataset.mode=cameraMode;
  cameraButton.setAttribute('aria-label',cameraDescriptions[cameraMode]);
  cameraButton.setAttribute('aria-pressed',String(cameraMode!=='free'));
  cameraButton.textContent=cameraLabels[cameraMode];
  cameraButton.title=cameraDescriptions[cameraMode]+'. Нажмите, чтобы сменить режим';
};
controls.enabled=false;
updateCameraButton();
cameraButton.addEventListener('click',()=>{
  const previous=cameraMode;
  cameraMode=cameraModes[(cameraModes.indexOf(cameraMode)+1)%cameraModes.length];
  controls.enabled=cameraMode==='free';
  if(controls.enabled){
    // Retain the visible camera position on transition to orbit mode.
    controls.target.copy(previous==='torso'?torsoPosition:followTarget);
    camera.up.set(0,0,1);
    controls.update();
  }
  updateCameraButton();
});
function updateFollowCamera(dt){
  // World-facing mode keeps the horizon fixed and follows the robot position.
  camera.up.set(0,0,1);
  targetGoal.set(rootPosition.x,rootPosition.y,
    THREE.MathUtils.clamp(rootPosition.z*.65+.18,.34,.86));
  const factor=1-Math.exp(-6*Math.min(dt,.1));
  followTarget.lerp(targetGoal,factor);
  eyeGoal.copy(followTarget).add(frontalOffset);
  camera.position.lerp(eyeGoal,factor);
  camera.lookAt(followTarget);
  canvas.dataset.cameraTargetZ=followTarget.z.toFixed(4);
  canvas.dataset.cameraTargetX=followTarget.x.toFixed(4);
}
function updateTorsoCamera(){
  // Track torso yaw only, never its pitch or roll.
  // The lens and its look-at point have the same world Z, keeping
  // the viewing axis EXACTLY parallel to the horizontal ground plane.
  // Keep a minimum eye height when the torso lies close to the floor.
  const z=Math.max(0.34,torsoPosition.z);
  horizontalLookAt.set(torsoPosition.x,torsoPosition.y,z);
  camera.position.copy(horizontalLookAt).addScaledVector(torsoHeading,cameraDistance);
  camera.up.copy(worldUp);
  camera.lookAt(horizontalLookAt);
  camera.getWorldDirection(cameraForward);
  canvas.dataset.cameraHorizontalZ=cameraForward.z.toFixed(6);
  canvas.dataset.cameraUpAlignment=camera.up.dot(worldUp).toFixed(6);
  canvas.dataset.cameraNormalAlignment=(-cameraForward.dot(torsoHeading)).toFixed(6);
  canvas.dataset.cameraTorsoNormalZ=torsoNormal.z.toFixed(4);
  canvas.dataset.cameraTargetZ=horizontalLookAt.z.toFixed(4);
  canvas.dataset.cameraTargetX=horizontalLookAt.x.toFixed(4);
}
scene.add(new THREE.HemisphereLight(0xb8d5f1,0x3b5363,2));
const key=new THREE.DirectionalLight(0xffffff,3);
key.position.set(3,-4,7);scene.add(key);
const grid=new THREE.GridHelper(12,30,0x43627d,0x284057);
grid.rotation.x=Math.PI/2;grid.position.z=-.01;scene.add(grid);
const platform=new THREE.Mesh(new THREE.PlaneGeometry(100,100),
  new THREE.MeshStandardMaterial({color:0x192938,roughness:1}));
platform.position.z=-.019;scene.add(platform);
let meshes=[],lastTime=performance.now(),accumulator=0,enabled=false,playing=false,workerReady=false;
let activeMotion='short',activeRepetitions=0;
const presetValues={
  short:[[300000,'300 тыс.'],[500000,'500 тыс.'],[850000,'850 тыс.']],
  full:[[10000,'10 тыс.'],[50000,'50 тыс.'],[125000,'125 тыс.']],
  walk:[[500000,'500 тыс.'],[1000000,'1 млн'],[1500000,'1,5 млн']]
};
function syncPresetSelection(){
  const chosen=Number($('repeats').value);
  document.querySelectorAll('#preset-buttons button').forEach(b=>{
    b.setAttribute('aria-pressed',String(Number(b.dataset.repeats)===chosen));
  });
}
function startRun(){
  if(!workerReady||!enabled)return;
  const R=Number($('repeats').value);
  if(!Number.isSafeInteger(R)||R<0||R>100000000){
    $('message').textContent='R должно быть целым числом от 0 до 100 млн';
    return;
  }
  activeMotion=$('motion').value;
  activeRepetitions=R;
  playing=false;
  accumulator=0;
  $('play').textContent='Запустить';
  $('play').disabled=true;
  $('verdict').textContent='Подготовка прогона';
  worker.postMessage({type:'start',motion:$('motion').value,repetitions:R});
}
function populatePresets(){
  const mode=$('motion').value,root=$('preset-buttons');
  root.replaceChildren();
  for(const [value,label] of presetValues[mode]){
    const btn=document.createElement('button');
    btn.type='button';
    btn.textContent=label;
    btn.dataset.repeats=String(value);
    btn.title=value.toLocaleString('ru-RU')+' условных повторений';
    btn.setAttribute('aria-pressed','false');
    btn.addEventListener('click',()=>{
      $('repeats').value=String(value);
      syncPresetSelection();
      startRun();
    });
    root.append(btn);
  }
  syncPresetSelection();
}
populatePresets();
$('repeats').addEventListener('input',()=>{
  syncPresetSelection();
  if(workerReady&&Number($('repeats').value)!==activeRepetitions){
    $('play').textContent='Запустить';
  }
});
const matrix=new THREE.Matrix4();
function resize(){
  const width=Math.max(100,canvas.clientWidth),height=Math.max(100,canvas.clientHeight);
  renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();
}
const observer=new ResizeObserver(resize);observer.observe(canvas);resize();
function updateTransform(msg){
  if(msg.root?.length===3&&msg.root.every(Number.isFinite)){
    rootPosition.set(...msg.root);
  }
  if(msg.torsoPosition?.length===3&&msg.torsoRotation?.length===9&&
      msg.torsoPosition.every(Number.isFinite)&&
      msg.torsoRotation.every(Number.isFinite)){
    torsoPosition.set(...msg.torsoPosition);
    const m=msg.torsoRotation;
    // Column 0 is torso-forward (+X), column 2 is torso-up (+Z).
    // MuJoCo's xmat is row-major, so extract each world-space column.
    torsoNormal.set(m[0],m[3],m[6]).normalize();
    // Project the chest normal on the ground plane. As it approaches
    // vertical during a fall, retain the last unambiguous yaw heading.
    const horizontalLength=Math.hypot(torsoNormal.x,torsoNormal.y);
    if(horizontalLength>TORSO_HEADING_MIN){
      torsoHeading.set(torsoNormal.x/horizontalLength,
        torsoNormal.y/horizontalLength,0);
    }
  }
  const x=msg.transform;
  if(x.length!==meshes.length*12)return;
  for(let i=0;i<meshes.length;i++){
    const o=i*12;
    matrix.set(x[o+3],x[o+4],x[o+5],x[o],
               x[o+6],x[o+7],x[o+8],x[o+1],
               x[o+9],x[o+10],x[o+11],x[o+2],0,0,0,1);
    meshes[i].matrix.copy(matrix);
    meshes[i].matrixWorldNeedsUpdate=true;
  }
  $('time').textContent=msg.time.toFixed(2)+' с';
}
function addGeometry(parts){
  for(const mesh of meshes){scene.remove(mesh);mesh.geometry.dispose();mesh.material.dispose()}
  meshes=parts.map(p=>{
    const indexed=new THREE.BufferGeometry();
    indexed.setAttribute('position',new THREE.Float32BufferAttribute(p.vertices,3));
    indexed.setIndex(new THREE.BufferAttribute(p.faces,1));
    const geometry=toCreasedNormals(indexed,THREE.MathUtils.degToRad(50));
    if(geometry!==indexed)indexed.dispose();
    const color=new THREE.Color().setRGB(p.rgba[0],p.rgba[1],p.rgba[2]);
    const mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({
      color,metalness:.23,roughness:.52,side:THREE.DoubleSide
    }));
    mesh.matrixAutoUpdate=false;scene.add(mesh);return mesh;
  });
}
const worker=new Worker(new URL('./engine.worker.js',import.meta.url),{type:'module'});
const base=new URL('./',document.baseURI).href;
worker.onmessage=({data:m})=>{
  if(m.type==='status'){$('message').textContent=m.message;$('stage-status').textContent=m.message}
  if(m.type==='ready'){
    addGeometry(m.geoms);enabled=true;workerReady=true;
    $('play').disabled=false;$('reset').disabled=false;
    $('stage-status').textContent='Модель G1 готова';
    // Opening the demo immediately starts the healthy (R=0) selected policy.
    $('repeats').value='0';
    syncPresetSelection();
    startRun();
  }
  if(m.type==='pose')updateTransform(m);
  if(m.type==='metrics'){
    $('health').textContent=m.health.toFixed(3);
    $('torque').textContent=m.torque.toFixed(3);
    $('weakest').textContent=m.weakest;
  }
  if(m.type==='playing'){
    playing=!m.paused;$('play').textContent=playing?'Пауза':'Продолжить';
    $('verdict').textContent=playing?'Выполняется':'Пауза';
    $('play').disabled=false;
  }
  if(m.type==='fall'){
    // Keep the animation loop active while unpowered MuJoCo is settling.
    playing=true;
    $('play').disabled=true;
    $('play').textContent='Свободное падение';
    $('verdict').textContent='Падение: приводы отключены';
    $('stage-status').textContent='Пассивная динамика, момент 0 Н·м';
  }
  if(m.type==='settled'){
    playing=false;
    $('play').disabled=false;
    $('play').textContent='Запустить';
    $('verdict').textContent=m.settled?'Лежит: управление отключено':'Пассивное падение: расчёт завершён';
    $('stage-status').textContent=m.settled?
      'Робот в покое. Для нового опыта нажмите «Сброс»':
      'Пассивный этап завершён по лимиту времени';
    $('message').textContent=m.settled?
      'Приводы остаются выключенными, физическая поза сохранена до сброса':
      'Суставы без управляющего момента. Конечная поза сохранена до сброса';
  }
  if(m.type==='reset'){
    playing=false;$('play').textContent='Запустить';
    $('time').textContent='0.00 с';$('verdict').textContent='Сброшено';
    $('health').textContent='1.000';$('torque').textContent='1.000';
    $('weakest').textContent='—';
  }
  if(m.type==='finish'){
    playing=false;$('play').textContent='Запустить';
    $('verdict').textContent=m.failed?'Падение':'Движение завершено';
  }
  if(m.type==='error'){
    playing=false;$('play').disabled=false;
    $('message').textContent='Ошибка: '+m.message;
    $('stage-status').textContent='Ошибка моделирования';
    console.error(m.message);
  }
};
worker.onerror=event=>{
  playing=false;$('stage-status').textContent='Ошибка WebAssembly/Worker';
  $('message').textContent=event.message;
};
$('play').addEventListener('click',()=>{
  if(!workerReady||!enabled)return;
  if(Number($('repeats').value)!==activeRepetitions||$('motion').value!==activeMotion){
    startRun();
    return;
  }
  if(playing||$('play').textContent==='Продолжить'){
    worker.postMessage({type:'pause'});
    return;
  }
  startRun();
});
$('reset').addEventListener('click',()=>{
  playing=false;
  accumulator=0;
  $('repeats').value='0';
  syncPresetSelection();
  $('play').disabled=false;
  worker.postMessage({type:'reset'});
});
$('motion').addEventListener('change',()=>{
  $('repeats').value='0';
  populatePresets();
  // A new policy starts in the healthy state without another user action.
  startRun();
});
function animate(now){
  requestAnimationFrame(animate);
  const elapsed=Math.min(.1,(now-lastTime)/1000);lastTime=now;
  if(cameraMode==='follow')updateFollowCamera(elapsed);
  else if(cameraMode==='torso')updateTorsoCamera();
  else controls.update();
  if(playing){accumulator+=elapsed; if(accumulator>=.02){accumulator-=.02;worker.postMessage({type:'tick'})}}
  renderer.render(scene,camera);
}
requestAnimationFrame(animate);
worker.postMessage({type:'init',base,cpu:new URL(location.href).searchParams.get('cpu')});
