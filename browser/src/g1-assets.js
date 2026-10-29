import loadMujoco from '@mujoco/mujoco';
import wasmUrl from '@mujoco/mujoco/mujoco.wasm?url';
// A single gzip archive replaces 37 separate STL requests (17.7 MiB).
// Its content-addressed URL lets CacheStorage survive page reloads safely.
async function fetchModelBundle(base,report){
  const root=new URL('model/',base);
  const metadataResponse=await fetch(new URL('asset-bundle.json',root),{cache:'no-cache'});
  if(!metadataResponse.ok)throw Error('G1 bundle manifest: HTTP '+metadataResponse.status);
  const info=await metadataResponse.json();
  if(!/^g1-assets-[a-f0-9]{16}[.]tar[.]gz$/.test(info.archive)||
      !Array.isArray(info.files)||!info.files.includes('g1.xml'))
    throw Error('Invalid G1 asset-bundle.json');
  const url=new URL(info.archive,root);
  let packed;
  let cached=false;
  let cache=null;
  if(typeof caches!=='undefined'){
    try{
      cache=await caches.open('wearbench-g1-meshes-v1');
      const response=await cache.match(url.href);
      if(response){packed=await response.arrayBuffer();cached=true;}
    }catch(e){console.warn('Model disk cache unavailable:',e);}
  }
  if(!packed){
    report('Загрузка сжатой модели G1');
    const response=await fetch(url);
    if(!response.ok)throw Error('G1 model archive: HTTP '+response.status);
    packed=await response.arrayBuffer();
    // Caching is best-effort, not part of the critical startup path.
    if(cache)void cache.put(url.href,new Response(packed.slice(0))).catch(e=>
      console.warn('G1 cache write failed:',e));
  }
  report((cached?'Кэш модели G1':'Сжатая модель G1')+
         ' · '+(packed.byteLength/1048576).toFixed(1)+' МиБ');
  if(typeof DecompressionStream==='undefined')
    throw Error('Browser does not support gzip DecompressionStream');
  const stream=new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip'));
  const data=new Uint8Array(await new Response(stream).arrayBuffer());
  return {data,files:info.files,cached,downloadBytes:packed.byteLength};
}
function installTarModel(mj,bytes,files){
  const decoder=new TextDecoder();
  const read=(start,length)=>{
    const block=bytes.subarray(start,start+length);
    const nul=block.indexOf(0);
    return decoder.decode(nul<0?block:block.subarray(0,nul)).trim();
  };
  const expected=new Set(files);
  let offset=0;
  const extracted=new Set();
  while(offset+512<=bytes.length){
    const name=read(offset,100);
    if(!name)break;
    const size=parseInt(read(offset+124,12),8);
    if(!Number.isSafeInteger(size)||size<0||offset+512+size>bytes.length)
      throw Error('Invalid model TAR entry: '+name);
    const type=bytes[offset+156];
    if(type===0||type===48){
      if(!expected.has(name)||extracted.has(name))throw Error('Unexpected model mesh: '+name);
      mj.FS.writeFile('/g1/'+name,bytes.subarray(offset+512,offset+512+size));
      extracted.add(name);
    }
    offset+=512+Math.ceil(size/512)*512;
  }
  if(extracted.size!==expected.size)throw Error(
    'Incomplete G1 model: '+extracted.size+' / '+expected.size+' assets');
}
export async function loadG1(base,report) {
  const started=performance.now();
  // Download the archive while WASM loads and compiles, not afterward.
  const archivePromise=fetchModelBundle(base,report);
  report('Загрузка MuJoCo WASM и модели G1');
  const mjPromise=loadMujoco({locateFile:n=>n.endsWith('.wasm')?wasmUrl:n})
    .catch(e=>{throw Error('MuJoCo WASM: '+(e?.message||String(e)))});
  const [mj,bundle]=await Promise.all([
    mjPromise,
    archivePromise.catch(e=>{throw Error('G1 assets: '+(e?.message||String(e)))})
  ]);
  mj.FS.mkdir('/g1');mj.FS.mkdir('/g1/meshes');
  installTarModel(mj,bundle.data,bundle.files);
  report('Геометрия G1 готова. Сборка MuJoCo...');
  let xml=new TextDecoder().decode(mj.FS.readFile('/g1/g1.xml'));
  xml=xml.replace('<worldbody>','<worldbody><geom name="ground" type="plane" pos="0 0 0" size="20 20 0.1" rgba=".1 .16 .22 1"/>');
  xml=xml.replace('<mujoco model="g1_29dof">','<mujoco model="g1_29dof"><option timestep="0.002" gravity="0 0 -9.81"/>');
  mj.FS.writeFile('/g1/g1.xml',new TextEncoder().encode(xml));
  const model=mj.MjModel.from_xml_path('/g1/g1.xml'), data=new mj.MjData(model);
  if(model.nu!==29||model.nq!==36||model.nv!==35)throw Error('Expected 29-DoF G1');
  const actuator=[];
  for(let i=0;i<29;i++){
    const jid=model.actuator_trnid[i*2];
    actuator.push({name:mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT.value,jid),
      q:model.jnt_qposadr[jid],v:model.jnt_dofadr[jid],
      lo:model.actuator_ctrlrange[i*2],hi:model.actuator_ctrlrange[i*2+1]});
  }
  const body=n=>mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY.value,n);
  const sensor=n=>mj.mj_name2id(model,mj.mjtObj.mjOBJ_SENSOR.value,n);
  const pelvis=body('pelvis'),torso=body('torso_link'),gyro=sensor('imu_gyro');
  if(Math.min(pelvis,torso,gyro)<0)throw Error('G1 scene lacks required body or IMU sensor');
  const visible=[],geoms=[],transfer=[];
  for(let id=0;id<model.ngeom;id++){
    if(model.geom_group[id]!==1||model.geom_type[id]!==7)continue;
    const mesh=model.geom_dataid[id],v=model.mesh_vertadr[mesh],vn=model.mesh_vertnum[mesh];
    const f=model.mesh_faceadr[mesh],fn=model.mesh_facenum[mesh];
    const vertices=Float32Array.from(model.mesh_vert.subarray(v*3,(v+vn)*3));
    const faces=Uint32Array.from(model.mesh_face.subarray(f*3,(f+fn)*3));
    const rgba=Float32Array.from(model.geom_rgba.subarray(id*4,id*4+4));
    geoms.push({vertices,faces,rgba});visible.push(id);
    transfer.push(vertices.buffer,faces.buffer,rgba.buffer);
  }
  const pose=(t,failed=false)=>{
    const x=new Float64Array(visible.length*12);
    visible.forEach((id,i)=>{
      x.set(data.geom_xpos.subarray(id*3,id*3+3),i*12);
      x.set(data.geom_xmat.subarray(id*9,id*9+9),i*12+3);
    });
    postMessage({
      type:'pose',time:t,failed,
      root:[data.qpos[0],data.qpos[1],data.qpos[2]],
      // MuJoCo body orientation is a row-major 3x3 matrix in world coordinates.
      // Use the current torso body frame, not the free-base quaternion.
      torsoPosition:Array.from(data.xipos.subarray(torso*3,torso*3+3)),
      torsoRotation:Array.from(data.xmat.subarray(torso*9,torso*9+9)),
      transform:x
    },[x.buffer]);
  };
  report('Модель G1 готова за '+((performance.now()-started)/1000).toFixed(1)+' с');
  return {mj,model,data,actuator,pelvis,torso,gyroAdr:model.sensor_adr[gyro],geoms,transfer,pose};
}
