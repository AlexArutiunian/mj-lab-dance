import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {toCreasedNormals} from 'three/addons/utils/BufferGeometryUtils.js';
import './style.css';
const $=name=>document.getElementById(name);
const canvas=$('viewport');
const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(42,1,0.02,200);
camera.up.set(0,0,1);camera.position.set(2.5,-3.6,1.8);
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
renderer.outputColorSpace=THREE.SRGBColorSpace;
canvas.appendChild(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);
controls.target.set(0,0,.88);controls.enableDamping=true;controls.update();
scene.add(new THREE.HemisphereLight(0xb8d5f1,0x3b5363,2));
const key=new THREE.DirectionalLight(0xffffff,3);
key.position.set(3,-4,7);scene.add(key);
const grid=new THREE.GridHelper(12,30,0x43627d,0x284057);
grid.rotation.x=Math.PI/2;grid.position.z=-.01;scene.add(grid);
const platform=new THREE.Mesh(new THREE.PlaneGeometry(100,100),
  new THREE.MeshStandardMaterial({color:0x192938,roughness:1}));
platform.position.z=-.019;scene.add(platform);
let meshes=[],lastTime=performance.now(),accumulator=0,enabled=false,playing=false,workerReady=false;
const matrix=new THREE.Matrix4();
function resize(){
  const width=Math.max(100,canvas.clientWidth),height=Math.max(100,canvas.clientHeight);
  renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();
}
const observer=new ResizeObserver(resize);observer.observe(canvas);resize();
function updateTransform(msg){
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
  if(playing){worker.postMessage({type:'pause'});return}
  if($('play').textContent==='Продолжить'){
    worker.postMessage({type:'pause'});return;
  }
  const R=Number($('repeats').value);
  if(!Number.isSafeInteger(R)||R<0){$('message').textContent='R должно быть целым числом от 0';return}
  $('play').disabled=true;$('verdict').textContent='Подготовка прогона';
  worker.postMessage({type:'start',motion:$('motion').value,repetitions:R});
});
$('reset').addEventListener('click',()=>{
  $('repeats').value='0';$('play').disabled=false;
  worker.postMessage({type:'reset'});
});
$('motion').addEventListener('change',()=>{
  playing=false;$('play').textContent='Запустить';
  worker.postMessage({type:'reset'});
});
function animate(now){
  requestAnimationFrame(animate);
  const elapsed=Math.min(.1,(now-lastTime)/1000);lastTime=now;
  controls.update();
  if(playing){accumulator+=elapsed; if(accumulator>=.02){accumulator-=.02;worker.postMessage({type:'tick'})}}
  renderer.render(scene,camera);
}
requestAnimationFrame(animate);
worker.postMessage({type:'init',base});
