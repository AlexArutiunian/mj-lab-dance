import {chromium} from 'playwright';
const browser=await chromium.launch({headless:true,args:['--enable-unsafe-swiftshader','--disable-dev-shm-usage']});
const page=await browser.newPage({viewport:{width:1200,height:750}});
const errors=[];
page.on('pageerror',e=>errors.push('page '+e.message));
page.on('console',m=>{if(m.type()==='error')errors.push('console '+m.text());});
page.on('requestfailed',q=>errors.push('network '+q.url()+' '+q.failure()?.errorText));
page.on('response',r=>{if(r.status()>=400)errors.push('HTTP '+r.status()+' '+r.url());});
try{
  await page.goto('http://127.0.0.1:4173/',{waitUntil:'domcontentloaded',timeout:30000});
  await page.waitForFunction(()=>!document.querySelector('#play').disabled,null,{timeout:120000});
  console.log('G1 model ready');
  // Follow mode is the default, with a front-facing camera and free-orbit toggle.
  const cameraMode=page.locator('#camera-mode');
  await page.waitForFunction(()=>Number.isFinite(
    Number(document.querySelector('#viewport').dataset.cameraTargetZ)),null,{timeout:10000});
  if(await cameraMode.getAttribute('data-mode')!=='follow')
    throw new Error('Expected follow camera mode on load');
  const initialLookZ=Number(await page.locator('#viewport').getAttribute('data-camera-target-z'));
  await cameraMode.click();
  if(await cameraMode.getAttribute('data-mode')!=='torso')
    throw new Error('Torso camera mode toggle failed');
  await page.waitForFunction(() =>
    Number(document.querySelector('#viewport').dataset.cameraNormalAlignment)>.99999,
    null,{timeout:10000});
  const cameraHorizontalZ=Number(await page.locator('#viewport').getAttribute('data-camera-horizontal-z'));
  const cameraUp=Number(await page.locator('#viewport').getAttribute('data-camera-up-alignment'));
  if(Math.abs(cameraHorizontalZ)>1e-6||cameraUp<.999999)
    throw new Error('Torso camera must remain horizontal with upright horizon');
  console.log('TORSO CAMERA: viewing direction parallel to floor, up aligned with gravity');
  await cameraMode.click();
  if(await cameraMode.getAttribute('data-mode')!=='free')
    throw new Error('Free orbit mode toggle failed');
  await cameraMode.click();
  if(await cameraMode.getAttribute('data-mode')!=='follow')
    throw new Error('Front-follow camera mode toggle failed');
  console.log('FRONT FOLLOW CAMERA: initial target Z',initialLookZ.toFixed(3));
  for(const mode of ['short','full','clip10']){
    await page.locator('#motion').selectOption(mode);
    await page.locator('#repeats').fill('0');
    await page.locator('#play').click();
    await page.waitForFunction(()=>parseFloat(document.querySelector('#time').textContent)>0.1,
      null,{timeout:120000});
    console.log('HEALTHY',mode,await page.locator('#time').textContent());
    await page.locator('#reset').click();
    await page.waitForTimeout(300);
  }
  await page.locator('#motion').selectOption('short');
  await page.locator('#repeats').fill('500000');
  await page.locator('#play').click();
  await page.waitForFunction(()=>parseFloat(document.querySelector('#time').textContent)>0.1 &&
    parseFloat(document.querySelector('#health').textContent)<1,null,{timeout:180000});
  console.log('AGED short: time',await page.locator('#time').textContent(),
    'health',await page.locator('#health').textContent(),
    'torque',await page.locator('#torque').textContent());
  // Under severe modelled torque loss, the robot must fall, switch off
  // active motor torques and keep stepping MuJoCo until the passive phase ends.
  await page.locator('#reset').click();
  await page.waitForTimeout(300);
  await page.locator('#motion').selectOption('short');
  await page.locator('#repeats').fill('100000000');
  // Watch the same modelled fall through its rotating torso frame.
  await cameraMode.click();
  if(await cameraMode.getAttribute('data-mode')!=='torso')
    throw new Error('Torso camera not selected for fall test');
  const standingTorsoNormalZ=Number(await page.locator('#viewport').getAttribute('data-camera-torso-normal-z'));
  const standingTorsoTargetZ=Number(await page.locator('#viewport').getAttribute('data-camera-target-z'));
  await page.locator('#play').click();
  await page.waitForFunction(() =>
    document.querySelector('#verdict').textContent.includes('приводы отключены'),
    null,{timeout:180000});
  const fallTime=Number.parseFloat(await page.locator('#time').textContent());
  console.log('FALL DETECTED at',fallTime,'s. ONNX/PD commands disabled');
  await page.waitForFunction(() =>
    document.querySelector('#verdict').textContent.includes('управление отключено') ||
    document.querySelector('#verdict').textContent.includes('расчёт завершён'),
    null,{timeout:120000});
  const afterFallTime=Number.parseFloat(await page.locator('#time').textContent());
  if(afterFallTime-fallTime<1.0)throw new Error('Passive MuJoCo did not advance after falling');
  console.log('PASSIVE FALL: simulation continued',Number(afterFallTime-fallTime).toFixed(2),'s');
  // Exact torso-normal tracking must remain perpendicular throughout falling.
  await page.waitForTimeout(350);
  const aligned=Number(await page.locator('#viewport').getAttribute('data-camera-normal-alignment'));
  const lyingTorsoNormalZ=Number(await page.locator('#viewport').getAttribute('data-camera-torso-normal-z'));
  const lyingTorsoTargetZ=Number(await page.locator('#viewport').getAttribute('data-camera-target-z'));
  if(!Number.isFinite(aligned)||aligned<.99999)
    throw new Error('Torso camera lost perpendicular alignment: '+aligned);
  if(!Number.isFinite(lyingTorsoTargetZ)||lyingTorsoTargetZ>=standingTorsoTargetZ-.12)
    throw new Error('Torso camera did not follow the actual falling body');
  if(!Number.isFinite(lyingTorsoNormalZ)||Math.abs(lyingTorsoNormalZ-standingTorsoNormalZ)<.05)
    throw new Error('Torso camera received no meaningful rotation during the fall');
  console.log('TORSO CAMERA FALL: exact normal alignment',aligned,
    'body axis Z',standingTorsoNormalZ.toFixed(3),'->',lyingTorsoNormalZ.toFixed(3),
    'target Z',standingTorsoTargetZ.toFixed(3),'->',lyingTorsoTargetZ.toFixed(3));
  // Confirm world-front mode continues to track the low post-fall position.
  await cameraMode.click();
  if(await cameraMode.getAttribute('data-mode')!=='free')
    throw new Error('Could not return to free mode');
  await cameraMode.click();
  if(await cameraMode.getAttribute('data-mode')!=='follow')
    throw new Error('Could not return to fixed front mode');
  await page.waitForTimeout(900);
  const lyingLookZ=Number(await page.locator('#viewport').getAttribute('data-camera-target-z'));
  if(!Number.isFinite(lyingLookZ)||lyingLookZ>initialLookZ-.06)
    throw new Error('Front-follow camera did not lower its target after the robot fell');
  console.log('FRONT FOLLOW CAMERA: target lowered from',initialLookZ.toFixed(3),'to',lyingLookZ.toFixed(3));
  const finalTime=await page.locator('#time').textContent();
  await page.waitForTimeout(700);
  if(await page.locator('#time').textContent()!==finalTime)
    throw new Error('Robot should retain the terminal pose until Reset');
  await page.locator('#reset').click();
  await page.waitForFunction(() =>
    document.querySelector('#time').textContent==='0.00 с' &&
    document.querySelector('#health').textContent==='1.000',
    null,{timeout:10000});
  console.log('RESET after passive fall OK');
  await page.screenshot({path:'smoke.png',fullPage:true});
} catch(error){
  console.error('BROWSER SMOKE ERROR:',error.message);
  console.error('Status:',await page.locator('#stage-status').textContent());
  console.error('Message:',await page.locator('#message').textContent());
  console.error('Other errors:',errors.slice(0,30));
  await page.screenshot({path:'smoke-failed.png',fullPage:true});
  process.exitCode=1;
} finally{await browser.close();}
