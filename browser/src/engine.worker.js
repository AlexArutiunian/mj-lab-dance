import {loadG1} from './g1-assets.js';
import {loadPreset,reset,step,scales,configureONNX} from './controller.js';
let world,base,prepared=new Map(),profile=new Map(),referenceMax=0;
let current=null,token=0,running=false,busy=false,selected='short';
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
  for(let i=0;i<p.motion.count;i++){
    if(id!==token)return null;
    await step(world,p,i,state,null,work);
    if(i%100===0){
      notify('Профиль '+name+': '+Math.round((i+1)/p.motion.count*100)+'%');
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
  if(!['short','full','clip10'].includes(name)||!Number.isSafeInteger(R)||R<0)
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
  current={p,state,limits,frame:0,failed:false};
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
    if(c.frame>=c.p.motion.count){running=false;postMessage({type:'finish',failed:c.failed});return}
    c.failed ||=await step(world,c.p,c.frame,c.state,c.limits,null);
    c.frame++;
    world.pose(c.frame*0.02,c.failed);
    if(c.frame>=c.p.motion.count){running=false;postMessage({type:'finish',failed:c.failed})}
  }finally{busy=false}
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
