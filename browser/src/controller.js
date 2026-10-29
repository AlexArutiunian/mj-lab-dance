import {loadJsPolicy} from './js-inference.js';
let ortPromise=null,wasmBase=null,useJs=false;
async function getOrt(){
  if(!ortPromise)ortPromise=import('onnxruntime-web/wasm').then(ort=>{
    ort.env.wasm.numThreads=1;
    ort.env.wasm.wasmPaths=new URL('ort/',wasmBase).href;
    return ort;
  });
  return ortPromise;
}
import {DT,healthState,alphaRef,mul,inv,yawQuat,torsoQuat,rot6} from './model.js';
const PHYSICS_DT=0.002;
const loadJson=async url=>{const r=await fetch(url);if(!r.ok)throw Error(url+' : HTTP '+r.status);return r.json()};
async function gzJson(url){
  const r=await fetch(url);
  if(!r.ok)throw Error(url+' : HTTP '+r.status);
  const raw=new Uint8Array(await r.arrayBuffer());
  if(raw.length===0)throw Error('Empty motion reference: '+url);
  if(raw[0]===0x1f&&raw[1]===0x8b){
    const stream=new Blob([raw]).stream().pipeThrough(new DecompressionStream('gzip'));
    return JSON.parse(await new Response(stream).text());
  }
  // Some servers transparently decompress HTTP responses with Content-Encoding: gzip.
  return JSON.parse(new TextDecoder().decode(raw));
}
export async function loadPreset(base,name,report){
  const url=new URL('policies/'+name+'/',base);
  const walking=name==='walk';
  const cfg=await loadJson(new URL('deploy.json',url));
  const motion=walking?null:await gzJson(new URL('motion.json.gz',url));
  report('Загрузка ONNX: '+name);
  let session;
  if(useJs){
    report('Легкий JS-инференс исходной ONNX-политики: '+name);
    session=await loadJsPolicy(url);
  }else{
    const ort=await getOrt();
    const opts={executionProviders:['wasm']};
    if(!walking)opts.externalData=[{path:'policy.onnx.data',data:new URL('policy.onnx.data',url).href}];
    session=await ort.InferenceSession.create(new URL('policy.onnx',url).href,opts);
  }
  if(cfg.default_joint_pos.length!==29||cfg.stiffness.length!==29||
      cfg.damping.length!==29||cfg.actions.JointPositionAction.scale.length!==29||
      Math.abs(cfg.step_dt-DT)>1e-8)throw Error('Invalid Unitree G1 deploy configuration');
  if(!walking && (motion.joint_pos[0].length!==29||motion.joint_vel[0].length!==29))
    throw Error('Invalid mimic reference motion');
  const frames=walking?Math.round(60/DT):motion.count;
  return {cfg,motion,session,name,walking,frames,duration:frames*DT};
}
export function reset(runtime,preset){
  const {mj,model,data}=runtime;
  data.qpos.set(model.qpos0);
  data.qpos.set([0,0,0.8,1,0,0,0],0);
  data.qpos.set(preset.cfg.default_joint_pos,7);
  data.qvel.fill(0);data.ctrl.fill(0);mj.mj_forward(model,data);
  const q=Array.from(data.qpos);
  const real=torsoQuat(q.slice(3,7),q.slice(7));
  return {
    action:new Float32Array(29),
    orient:preset.walking?null:mul(yawQuat(real),inv(yawQuat(preset.motion.root_quat[0]))),
    phase:0,bad:0
  };
}
const cross=(a,b)=>[
  a[1]*b[2]-a[2]*b[1],
  a[2]*b[0]-a[0]*b[2],
  a[0]*b[1]-a[1]*b[0]
];
function inverseRotateGravity(quat){
  const [w,x,y,z]=quat;
  const norm=Math.hypot(w,x,y,z);
  const v=[x/norm,y/norm,z/norm],scalar=w/norm,g=[0,0,-1];
  const inner=cross(v,g).map((c,i)=>c-scalar*g[i]);
  const diff=cross(v,inner);
  return g.map((value,i)=>value+2*diff[i]);
}
export async function step(runtime,preset,frame,state,limits,work){
  const {mj,model,data,actuator,gyroAdr,pelvis,torso}=runtime;
  const cfg=preset.cfg;
  const joint=Array.from(data.qpos.subarray(7,36));
  let obs;
  if(preset.walking){
    // Native deploy equivalent: gyro, projected gravity, velocity command,
    // gait phase, qpos-default, qvel, previous action (98 floats).
    const period=cfg.observations.gait_phase.params.period;
    state.phase=(state.phase+DT/period)%1;
    obs=new Float32Array(98);
    obs.set(data.sensordata.subarray(gyroAdr,gyroAdr+3),0);
    obs.set(inverseRotateGravity(Array.from(data.qpos.subarray(3,7))),3);
    obs.set([0.4,0,0],6);
    obs.set([Math.sin(state.phase*2*Math.PI),Math.cos(state.phase*2*Math.PI)],9);
    for(let j=0;j<29;j++){
      obs[11+j]=joint[j]-cfg.default_joint_pos[j];
      obs[40+j]=data.qvel[6+j];
    }
    obs.set(state.action,69);
  } else {
    const ref=preset.motion,i=Math.min(frame,ref.count-1);
    const real=torsoQuat(Array.from(data.qpos.subarray(3,7)),joint);
    const desired=torsoQuat(ref.root_quat[i],ref.joint_pos[i]);
    obs=new Float32Array(154);
    obs.set(ref.joint_pos[i],0);obs.set(ref.joint_vel[i],29);
    obs.set(rot6(real,desired,state.orient),58);
    obs.set(data.sensordata.subarray(gyroAdr,gyroAdr+3),64);
    for(let j=0;j<29;j++){obs[67+j]=joint[j]-cfg.default_joint_pos[j];obs[96+j]=data.qvel[6+j]}
    obs.set(state.action,125);
  }
  const input=preset.session.inputNames[0],output=preset.session.outputNames[0];
  const feed=preset.session.backend==='js-fp32' ? {data:obs} :
    new (await getOrt()).Tensor('float32',obs,[1,obs.length]);
  const predicted=await preset.session.run({[input]:feed});
  const action=predicted[output].data;
  if(action.length!==29||!Array.from(action).every(Number.isFinite))throw Error('Invalid ONNX action');
  state.action.set(action);
  const targets=cfg.actions.JointPositionAction.scale.map((scale,j)=>
    scale*action[j]+cfg.actions.JointPositionAction.offset[j]);
  for(let sub=0;sub<10;sub++){
    for(let j=0;j<29;j++){
      const a=actuator[j],qIndex=a.q-7;
      const raw=cfg.stiffness[qIndex]*(targets[qIndex]-data.qpos[a.q])
        -cfg.damping[qIndex]*data.qvel[a.v];
      const scale=limits?.[j]??1;
      data.ctrl[j]=Math.max(a.lo*scale,Math.min(a.hi*scale,raw));
      if(work)work[j]+=Math.abs(data.ctrl[j]*data.qvel[a.v])*PHYSICS_DT;
    }
    mj.mj_step(model,data);
  }
  const low=data.xipos[pelvis*3+2]<0.45&&data.xipos[torso*3+2]<0.55;
  state.bad=low?state.bad+1:0;
  return state.bad>=Math.ceil(.5/DT);
}
/**
 * Passive fall dynamics: remove all actuator torques and integrate contacts
 * and gravity at the same 2 ms timestep as the controlled policy.
 * The policy and PD controller are not evaluated after failure detection.
 */
export function passiveStep(runtime){
  const {mj,data}=runtime;
  data.ctrl.fill(0);
  for(let sub=0;sub<10;sub++)mj.mj_step(runtime.model,data);
  let maxVelocity=0;
  for(let i=0;i<data.qvel.length;i++){
    const v=data.qvel[i];
    if(!Number.isFinite(v))throw new Error('Unstable MuJoCo passive dynamics');
    maxVelocity=Math.max(maxVelocity,Math.abs(v));
  }
  return {maxVelocity, contacts:data.ncon};
}
export function scales(profile,R,refMax){
  const kappa=alphaRef/refMax;
  const state=Array.from(profile,(S)=>healthState(R,S,kappa));
  return {limits:state.map(s=>s.u),state};
}
export function configureONNX(base,cpu){
  wasmBase=base;
  // Safari on 3 GB iPhones cannot reliably hold MuJoCo WASM and ORT WASM
  // simultaneously. Run the same ONNX graph and fp32 weights with JS instead.
  useJs=cpu==='js'||/iPhone|iPad|iPod/.test(self.navigator?.userAgent||'')||
    (self.navigator?.platform==='MacIntel'&&self.navigator?.maxTouchPoints>1);
}
