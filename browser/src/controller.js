import * as ort from 'onnxruntime-web';
import {DT,healthState,alphaRef,mul,inv,yawQuat,torsoQuat,rot6} from './model.js';
const PHYSICS_DT=0.002;
const loadJson=async url=>{const r=await fetch(url);if(!r.ok)throw Error(url+' : HTTP '+r.status);return r.json()};
async function gzJson(url){
  const r=await fetch(url);if(!r.ok)throw Error(url+' : HTTP '+r.status);
  return JSON.parse(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).text());
}
export async function loadPreset(base,name,report){
  const url=new URL('policies/'+name+'/',base);
  const [cfg,motion]=await Promise.all([loadJson(new URL('deploy.json',url)),gzJson(new URL('motion.json.gz',url))]);
  report('Загрузка ONNX: '+name);
  const session=await ort.InferenceSession.create(new URL('policy.onnx',url).href,{
    executionProviders:['wasm'],
    externalData:[{path:'policy.onnx.data',data:new URL('policy.onnx.data',url).href}]
  });
  if(cfg.default_joint_pos.length!==29||motion.joint_pos[0].length!==29)throw Error('Bad 29-DoF preset');
  return {cfg,motion,session,name};
}
export function reset(runtime,preset){
  const {mj,model,data}=runtime;
  data.qpos.set(model.qpos0);
  data.qpos.set([0,0,0.8,1,0,0,0],0);
  data.qpos.set(preset.cfg.default_joint_pos,7);
  data.qvel.fill(0);data.ctrl.fill(0);mj.mj_forward(model,data);
  const q=Array.from(data.qpos);
  const real=torsoQuat(q.slice(3,7),q.slice(7));
  return {action:new Float32Array(29),orient:mul(yawQuat(real),inv(yawQuat(preset.motion.root_quat[0]))),bad:0};
}
export async function step(runtime,preset,frame,state,limits,work){
  const {mj,model,data,actuator,gyroAdr,pelvis,torso}=runtime;
  const cfg=preset.cfg,ref=preset.motion,i=Math.min(frame,ref.count-1);
  const joint=Array.from(data.qpos.subarray(7,36));
  const real=torsoQuat(Array.from(data.qpos.subarray(3,7)),joint);
  const desired=torsoQuat(ref.root_quat[i],ref.joint_pos[i]);
  const obs=new Float32Array(154);
  obs.set(ref.joint_pos[i],0);obs.set(ref.joint_vel[i],29);
  obs.set(rot6(real,desired,state.orient),58);
  obs.set(data.sensordata.subarray(gyroAdr,gyroAdr+3),64);
  for(let j=0;j<29;j++){obs[67+j]=joint[j]-cfg.default_joint_pos[j];obs[96+j]=data.qvel[6+j]}
  obs.set(state.action,125);
  const input=preset.session.inputNames[0],output=preset.session.outputNames[0];
  const predicted=await preset.session.run({[input]:new ort.Tensor('float32',obs,[1,154])});
  const action=predicted[output].data;
  if(action.length!==29||!Array.from(action).every(Number.isFinite))throw Error('Invalid ONNX action');
  state.action.set(action);
  const targets=cfg.actions.JointPositionAction.scale.map((s,j)=>s*action[j]+cfg.actions.JointPositionAction.offset[j]);
  for(let sub=0;sub<10;sub++){
    for(let j=0;j<29;j++){
      const a=actuator[j],qIndex=a.q-7;
      const raw=cfg.stiffness[qIndex]*(targets[qIndex]-data.qpos[a.q])-cfg.damping[qIndex]*data.qvel[a.v];
      const s=limits?.[j]??1;
      data.ctrl[j]=Math.max(a.lo*s,Math.min(a.hi*s,raw));
      if(work)work[j]+=Math.abs(data.ctrl[j]*data.qvel[a.v])*PHYSICS_DT;
    }
    mj.mj_step(model,data);
  }
  const low=data.xipos[pelvis*3+2]<0.45&&data.xipos[torso*3+2]<0.55;
  state.bad=low?state.bad+1:0;
  return state.bad>=Math.ceil(.5/DT);
}
export function scales(profile,R,refMax){
  const kappa=alphaRef/refMax;
  const state=Array.from(profile,(S)=>healthState(R,S,kappa));
  return {limits:state.map(s=>s.u),state};
}
export function configureONNX(base){
  ort.env.wasm.numThreads=1;
  ort.env.wasm.wasmPaths=new URL('ort/',base).href;
}
