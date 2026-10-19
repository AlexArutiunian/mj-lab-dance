import loadMujoco from '@mujoco/mujoco';
import wasmUrl from '@mujoco/mujoco/mujoco.wasm?url';
export async function loadG1(base,report) {
  report('Загрузка MuJoCo WASM');
  const mj=await loadMujoco({locateFile:n=>n.endsWith('.wasm')?wasmUrl:n});
  mj.FS.mkdir('/g1');mj.FS.mkdir('/g1/meshes');
  const root=new URL('model/',base);
  const manifest=await (await fetch(new URL('manifest.json',root))).json();
  let index=0;
  await Promise.all(Array.from({length:4},async()=>{
    while(index<manifest.files.length){
      const file=manifest.files[index++];
      const response=await fetch(new URL(file,root));
      if(!response.ok)throw Error('Model asset missing: '+file);
      mj.FS.writeFile('/g1/'+file,new Uint8Array(await response.arrayBuffer()));
    }
  }));
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
    postMessage({type:'pose',time:t,failed,root:[data.qpos[0],data.qpos[1],data.qpos[2]],transform:x},[x.buffer]);
  };
  report('Модель G1 готова');
  return {mj,model,data,actuator,pelvis,torso,gyroAdr:model.sensor_adr[gyro],geoms,transfer,pose};
}
