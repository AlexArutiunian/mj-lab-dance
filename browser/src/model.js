export const DT=0.02, PHYSICS_DT=0.002;
export const floor=0.10, exponent=1.5;
export const alphaRef=(1-Math.pow((0.3-floor)/(1-floor),1/exponent))/1e6;
export function clamp(x,a=0,b=1){return Math.max(a,Math.min(b,x));}
export function healthState(R,work,kappa){
  if(R<0||work<0||kappa<0)throw new RangeError('nonnegative values required');
  const D=R*kappa*work,h=clamp(1-D),u=floor+(1-floor)*Math.pow(h,exponent);
  return {D,h,u};
}
export function mul(a,b){
const [w,x,y,z]=a,[v,p,q,r]=b;
return [w*v-x*p-y*q-z*r,w*p+x*v+y*r-z*q,w*q-x*r+y*v+z*p,w*r+x*q-y*p+z*v];
}
export function inv(q){const n=q.reduce((s,v)=>s+v*v,0);return [q[0]/n,-q[1]/n,-q[2]/n,-q[3]/n];}
export function axis(axis,angle){const q=[Math.cos(angle/2),0,0,0];q[axis+1]=Math.sin(angle/2);return q;}
export function yawQuat(q){const [w,x,y,z]=q;return axis(2,Math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)));}
export function torsoQuat(root,j){return mul(mul(mul(root,axis(2,j[12])),axis(0,j[13])),axis(1,j[14]));}
export function rot6(real,reference,initial){
const [w,x,y,z]=mul(inv(mul(initial,reference)),real);
return [1-2*(y*y+z*z),2*(x*y+z*w),2*(x*y-z*w),1-2*(x*x+z*z),2*(x*z+y*w),2*(y*z-x*w)];
}
