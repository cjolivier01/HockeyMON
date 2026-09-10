"""Self-contained browser UI for the loopback rink editor."""

PAGE = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Rink leveling and crop</title>
<style>
body{font:16px system-ui;background:#151a23;color:#eaf0f8;margin:24px auto;max-width:1300px;padding:0 16px}
button,input,select{font:inherit;margin:4px;padding:7px;border-radius:5px}button{cursor:pointer}
input[type=number]{width:88px}canvas{display:block;max-width:100%;height:auto;background:#080c12;touch-action:none;cursor:crosshair}
.controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:12px 0}fieldset{border:1px solid #526079;margin:16px 0}
#status{white-space:pre-wrap;min-height:1.5em;color:#9fe0ff}small{color:#bdc9d9}button:disabled{cursor:default;opacity:.5}
</style>
<h1>Rink leveling and crop</h1>
<p>Mark at least three vertical posts spread across the rink. Click each post's two endpoints
in its original camera view. Avoid horizontal rails and short marks. Preview the result before saving.</p>
<fieldset><legend>Vertical posts</legend>
<div class="controls"><select id="camera" aria-label="Camera"></select><button id="undo">Undo mark</button>
<button id="clear">Clear marks</button><span id="count"></span></div>
<canvas id="source"></canvas>
<div class="controls"><button id="estimate">Estimate pitch and roll</button><span id="quality"></span></div></fieldset>
<fieldset><legend>Leveling and panorama crop</legend>
<div class="controls"><label>Yaw <input id="yaw" type="number" min="-180" max="180" step="0.1"></label>
<label>Pitch <input id="pitch" type="number" min="-180" max="180" step="0.1"></label>
<label>Roll <input id="roll" type="number" min="-180" max="180" step="0.1"></label>
<button id="reset">Restore opening rotation</button><button id="zero">Zero rotation</button></div>
<small>Angles are degrees. Estimating preserves the yaw shown above.</small>
<div class="controls"><label><input id="auto" type="checkbox">Automatic crop</label>
<button id="full">Full canvas</button><button id="preview">Render preview</button></div>
<small>For a manual crop, drag a rectangle on the rendered panorama or enter percentages. Render again after changes.</small>
<div class="controls" id="bounds"><label>Left % <input id="left" type="number" min="0" max="100" step="0.1"></label>
<label>Right % <input id="right" type="number" min="0" max="100" step="0.1"></label>
<label>Top % <input id="top" type="number" min="0" max="100" step="0.1"></label>
<label>Bottom % <input id="bottom" type="number" min="0" max="100" step="0.1"></label></div>
<canvas id="panorama" hidden></canvas></fieldset>
<p id="status" role="status"></p>
<button id="save" disabled>Save for next calibration</button><button id="close">Close editor</button>
<p><small>Saving updates this game's config. Recalibrate and restart tracking to apply the new geometry.</small></p>
<script>
const $=id=>document.getElementById(id), angles=['yaw','pitch','roll'], bounds=['left','right','top','bottom'];
const auth=location.hash.slice(1);
let info, posts=[], pending=null, camera=0, sources=[], pano=null, previewToken=null, renderedCrop=null, drag=null, busy=false;
function status(text){$('status').textContent=text}
function invalidate(){previewToken=null;$('save').disabled=true;renderedCrop=null;drawPano()}
function state(){return {rotation_degrees:angles.map(id=>Number($(id).value)),crop:$('auto').checked?[0,1,0,1]:bounds.map(id=>Number($(id).value)/100),auto_crop:$('auto').checked,posts}}
function setAngles(values){angles.forEach((id,i)=>$(id).value=values[i]);invalidate()}
function setCrop(values){bounds.forEach((id,i)=>$(id).value=Number((100*values[i]).toFixed(4)));invalidate()}
async function request(path,data){const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Editor-Token':auth},body:JSON.stringify(data)});const value=await response.json();if(!response.ok)throw Error(value.error);return value}
async function action(fn){if(busy)return;busy=true;const controls=[...document.querySelectorAll('button,input,select')];const was=controls.map(el=>el.disabled);controls.forEach(el=>el.disabled=true);try{await fn()}catch(e){status(e.message)}finally{busy=false;controls.forEach((el,i)=>el.disabled=was[i]);$('save').disabled=!previewToken;bounds.forEach(id=>$(id).disabled=$('auto').checked)}}
function loadImage(url){return new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(Error('Unable to load preview image'));im.src=url})}
function point(event,canvas,width,height,cropEdges=false){const rect=canvas.getBoundingClientRect();return [Math.max(0,Math.min(cropEdges?width:width-1,(event.clientX-rect.left)/rect.width*width)),Math.max(0,Math.min(cropEdges?height:height-1,(event.clientY-rect.top)/rect.height*height))]}
function drawSource(){const canvas=$('source'),image=sources[camera];if(!image)return;canvas.width=image.width;canvas.height=image.height;const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);const size=info.sources[camera];ctx.scale(canvas.width/size.width,canvas.height/size.height);ctx.lineWidth=3*size.width/canvas.width;ctx.strokeStyle='#52f5a0';ctx.fillStyle='#52f5a0';posts.forEach((post,i)=>{if(post.image_index!==camera)return;ctx.beginPath();ctx.moveTo(...post.first);ctx.lineTo(...post.second);ctx.stroke();ctx.font=`${18*size.width/canvas.width}px sans-serif`;ctx.fillText(String(i+1),post.first[0]+8,post.first[1])});if(pending){ctx.beginPath();ctx.arc(...pending,5*size.width/canvas.width,0,Math.PI*2);ctx.fill()}$('count').textContent=`${posts.length} posts${pending?' — click second endpoint':''}`}
function drawPano(){if(!pano)return;const c=$('panorama');c.hidden=false;c.width=pano.width;c.height=pano.height;const ctx=c.getContext('2d');ctx.drawImage(pano,0,0);const crop=renderedCrop||state().crop;const [l,r,t,b]=crop;ctx.fillStyle='#0008';ctx.fillRect(0,0,c.width,t*c.height);ctx.fillRect(0,b*c.height,c.width,(1-b)*c.height);ctx.fillRect(0,t*c.height,l*c.width,(b-t)*c.height);ctx.fillRect(r*c.width,t*c.height,(1-r)*c.width,(b-t)*c.height);ctx.strokeStyle='#52f5a0';ctx.lineWidth=3;ctx.strokeRect(l*c.width,t*c.height,(r-l)*c.width,(b-t)*c.height)}
$('camera').onchange=()=>{camera=Number($('camera').value);pending=null;drawSource()};
$('source').onclick=event=>{if(busy)return;if(posts.length>=64&&!pending){status('At most 64 posts are supported. Undo a mark to replace it.');return}const size=info.sources[camera],p=point(event,$('source'),size.width,size.height);if(pending){posts.push({image_index:camera,first:pending,second:p});pending=null}else pending=p;invalidate();drawSource()};
$('undo').onclick=()=>{if(pending)pending=null;else posts.pop();invalidate();drawSource()};
$('clear').onclick=()=>{posts=[];pending=null;invalidate();drawSource()};
$('estimate').onclick=()=>action(async()=>{status('Converting camera points and fitting vertical posts…');const result=await request('/estimate',{posts,yaw:Number($('yaw').value)});setAngles(result.rotation_degrees);const rejected=posts.map((_,i)=>i).filter(i=>!result.inlier_indices.includes(i));$('quality').textContent=`${result.inlier_indices.length}/${posts.length} inliers; RMS ${result.rms_residual_degrees.toFixed(3)}°${rejected.length?'; excluded posts: '+rejected.map(i=>i+1).join(', '):''}`;status('Tilt estimated. Render a preview to check the result.')});
$('preview').onclick=()=>action(async()=>{invalidate();status('Rendering a bounded panorama preview…');const result=await request('/preview',state());pano=await loadImage('/preview/'+encodeURIComponent(result.token)+'.png');previewToken=result.token;renderedCrop=result.crop;drawPano();status(`Preview ready (${result.width} × ${result.height}). Save applies these settings on the next calibration.`)});
$('save').onclick=()=>action(async()=>{await request('/save',{state:state(),token:previewToken});previewToken=null;status('Saved. Recalibrate and restart tracking to apply the new geometry.');$('close').textContent='Close saved editor'});
$('close').onclick=()=>action(async()=>{await request('/close',{});previewToken=null;status('Editor closed. You can close this browser tab.')});
$('reset').onclick=()=>setAngles(info.rotation_degrees);$('zero').onclick=()=>setAngles([0,0,0]);
$('full').onclick=()=>{$('auto').checked=false;setCrop([0,1,0,1]);bounds.forEach(id=>$(id).disabled=false)};
$('auto').onchange=()=>{invalidate();bounds.forEach(id=>$(id).disabled=$('auto').checked)};
[...angles,...bounds].forEach(id=>$(id).oninput=invalidate);
$('panorama').onpointerdown=event=>{if(busy||$('auto').checked)return;drag=point(event,$('panorama'),pano.width,pano.height,true);$('panorama').setPointerCapture(event.pointerId)};
$('panorama').onpointerup=event=>{if(!drag||busy)return;const end=point(event,$('panorama'),pano.width,pano.height,true);const start=drag;drag=null;if(Math.abs(start[0]-end[0])<3||Math.abs(start[1]-end[1])<3)return;setCrop([Math.min(start[0],end[0])/pano.width,Math.max(start[0],end[0])/pano.width,Math.min(start[1],end[1])/pano.height,Math.max(start[1],end[1])/pano.height])};
action(async()=>{info=await request('/info',{});sources=await Promise.all(info.sources.map((source,i)=>{const option=document.createElement('option');option.value=i;option.textContent=`${i+1}: ${source.name} (${source.width} × ${source.height})`;$('camera').append(option);return loadImage('/source/'+i+'.jpg')}));setAngles(info.rotation_degrees);setCrop(info.crop);$('auto').checked=info.auto_crop;drawSource();status('Choose posts or adjust rotation/crop, then render a preview.')});
</script></html>"""
