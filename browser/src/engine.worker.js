import {loadG1} from './g1-assets.js';
import {loadPreset,reset,step,passiveStep,scales,configureONNX} from './controller.js';
let world,base,prepared=new Map(),profile=new Map(),referenceMax=0;
let current=null,token=0,running=false,busy=false,selected='short';
// All durations are simulation time at the 50 Hz control/update cadence.
const PASSIVE_MIN_FRAMES=100;       // at least 2 s of unpowered dynamics
const PASSIVE_STABLE_FRAMES=40;     // 0.8 s continuously at rest
const PASSIVE_MAX_FRAMES=1000;      // 20 s safety bound
const REST_VELOCITY_MAX=0.08;      // m/s and rad/s from MuJoCo generalized velocities
const notify=message=>postMessage({type:'status',message});
const microtask=()=>new Promise(r=>setTimeout(r,0));
async function init(msg){
  base=msg.base;
  configureONNX(base);
  world=await loadG1(base,notify);
  postMessage({type:'ready',geoms:world.geoms},world.transfer);
  world.data.qpos.set([0,0,0.8,1,0,0,0],0);
  world.mj.mj_forward(world.model,world.data);
  world.pose(0);
}
async function preset(name){
  if(!prepared.has(name))prepared.set(name,await loadPreset(base,name,notify));
  return prepared.get(name);
}
async function calculate(name,id){
  if(profile.has(name))return profile.get(name);
  if(name!=='short'&&!referenceMax)await calculate('short',id);
  if(id!==token)return null;
  const p=await preset(name);
  const state=reset(world,p),work=new Float64Array(29);
  notify('Расчёт базового профиля нагрузки: '+name);
  for(let i=0;i<p.frames;i++){
    if(id!==token)return null;
    await step(world,p,i,state,null,work);
    if(i%100===0){
      notify('Профиль '+name+': '+Math.round((i+1)/p.frames*100)+'%');
      await microtask();
    }
  }
  profile.set(name,work);
  if(name==='short')referenceMax=Math.max(...work);
  if(!(referenceMax>0))throw Error('Нулевой эталонный профиль нагрузки');
  return work;
}
async function start(msg){
  const id=++token;
  running=false;busy=false;current=null;
  const name=msg.motion,R=msg.repetitions;
  if(!['short','full','walk'].includes(name)||!Number.isSafeInteger(R)||R<0)
    throw Error('Некорректные параметры');
  selected=name;
  const p=await preset(name);
  let limits=Array(29).fill(1),statuses=null;
  if(R>0){
    const load=await calculate(name,id);
    if(id!==token||!load)return;
    const configured=scales(load,R,referenceMax);
    limits=configured.limits;statuses=configured.state;
  }
  if(id!==token)return;
  const state=reset(world,p);
  current={p,state,limits,frame:0,failed:false,phase:'policy',passiveFrames:0,stableFrames:0};
  running=true;
  const weakest=statuses?statuses.reduce((a,b)=>a.u<b.u?a:b):{h:1,u:1};
  const index=statuses?statuses.indexOf(weakest):-1;
  postMessage({type:'metrics',health:weakest.h,torque:weakest.u,
    weakest:index<0?'нет':world.actuator[index].name});
  world.pose(0);
  postMessage({type:'playing',paused:false});
  notify('Инференс ONNX / физика MuJoCo');
}
async function tick(){
  if(!running||busy||!current)return;
  busy=true;
  try{
    const c=current;
    if(c.phase==='policy'){
      if(c.frame>=c.p.frames){
        running=false;
        postMessage({type:'finish',failed:false});
        return;
      }
      // Always perform policy inference while healthy: do not short-circuit
      // subsequent MuJoCo steps when a failure has been detected.
      const fallen=await step(world,c.p,c.frame,c.state,c.limits,null);
      c.frame++;
      if(fallen){
        c.failed=true;
        c.phase='passive';
        world.data.ctrl.fill(0);
        postMessage({type:'fall',time:c.frame*0.02});
        notify('Падение зафиксировано. Моменты приводов обнулены, расчёт продолжается без управления');
      }
      world.pose(c.frame*0.02,c.failed);
      if(!c.failed&&c.frame>=c.p.frames){
        running=false;
        postMessage({type:'finish',failed:false});
      }
      return;
    }

    // Post-failure sequence: physics keeps stepping with all motor commands zero.
    // No ONNX inference, no target angles, and no PD control in this phase.
    if(c.phase==='passive'){
      const motion=passiveStep(world);
      c.passiveFrames++;
      const nearlyStatic=motion.contacts>0&&motion.maxVelocity<REST_VELOCITY_MAX;
      c.stableFrames=nearlyStatic?c.stableFrames+1:0;
      world.pose((c.frame+c.passiveFrames)*0.02,true);
      const settled=c.passiveFrames>=PASSIVE_MIN_FRAMES&&
        c.stableFrames>=PASSIVE_STABLE_FRAMES;
      const timedOut=c.passiveFrames>=PASSIVE_MAX_FRAMES;
      if(settled||timedOut){
        c.phase='rest';
        running=false;
        // Leave the final physical pose on the floor until explicit Reset.
        world.data.ctrl.fill(0);
        postMessage({type:'settled',settled,contacts:motion.contacts,
          maxVelocity:motion.maxVelocity,passiveTime:c.passiveFrames*0.02});
      }
    }
  }finally{busy=false;}
}
self.onmessage=async({data:m})=>{
  try{
    if(m.type==='init'){await init(m);return}
    if(m.type==='start'){await start(m);return}
    if(m.type==='tick'){await tick();return}
    if(m.type==='pause'){running=!running;postMessage({type:'playing',paused:!running});return}
    if(m.type==='reset'){
      token++;running=false;current=null;
      const p=prepared.get(selected);
      if(p)reset(world,p);
      if(world)world.pose(0);
      postMessage({type:'reset'});return;
    }
  }catch(e){
    running=false;
    postMessage({type:'error',message:e?.message||String(e)});
  }
};
