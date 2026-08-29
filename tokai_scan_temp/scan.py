from __future__ import annotations

import io, json, math, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'out'; CACHE=ROOT/'cache'
OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
S=requests.Session(); S.headers['User-Agent']='TokaiSurfaceScan/1.0'

REGIONS=[
 ('G01','岐阜','郡上美並北部・神路川上流支谷',(136.93,35.68,137.02,35.76)),
 ('G02','岐阜','郡上白鳥南部・牛道川側谷',(136.84,35.84,136.95,35.93)),
 ('G03','岐阜','根尾東部・根尾川支流',(136.58,35.65,136.68,35.75)),
 ('G04','岐阜','郡上八幡西部・栗巣川支谷',(136.84,35.71,136.94,35.80)),
 ('G05','岐阜','揖斐川町春日北部・山越え谷口',(136.43,35.49,136.53,35.59)),
 ('A01','愛知','東栄北部・大千瀬川支谷',(137.67,35.05,137.74,35.12)),
 ('M01','三重','大台町宮川上流・栗谷川合流外縁',(136.16,34.23,136.28,34.32)),
 ('M02','三重','大台町宮川中上流・側谷入口',(136.24,34.28,136.38,34.38)),
 ('M03','三重','尾鷲九鬼北西・湾奥背後',(136.20,34.04,136.30,34.13)),
 ('M04','三重','熊野北部・新鹿湾背後',(136.10,33.88,136.20,33.97)),
]

def ll2t(lon,lat,z):
 n=2**z; lat=math.radians(max(min(lat,85.05112878),-85.05112878))
 return (lon+180)/360*n,(1-math.asinh(math.tan(lat))/math.pi)/2*n

def t2ll(x,y,z):
 n=2**z
 return x/n*360-180,math.degrees(math.atan(math.sinh(math.pi*(1-2*y/n))))

def mpp(lat,z): return 156543.03392804097*math.cos(math.radians(lat))/(2**z)

def get(url,sfx):
 p=CACHE/(str(abs(hash(url)))+sfx)
 if p.exists(): return p.read_bytes()
 for k in range(4):
  try:
   r=S.get(url,timeout=25)
   if r.status_code==200 and len(r.content)>100:
    p.write_bytes(r.content); return r.content
   if r.status_code in (403,404): return None
  except requests.RequestException: pass
  time.sleep(.5*(k+1))
 return None

def dec_gsi(c):
 try: a=np.asarray(Image.open(io.BytesIO(c)).convert('RGB'),np.int32)
 except Exception: return None
 x=(a[:,:,0]<<16)+(a[:,:,1]<<8)+a[:,:,2]; h=np.empty(x.shape,np.float32)
 lo=x<(1<<23); hi=x>(1<<23); h[lo]=x[lo]*.01; h[hi]=(x[hi]-(1<<24))*.01; h[x==(1<<23)]=np.nan
 return h

def dec_q(c):
 try: a=np.asarray(Image.open(io.BytesIO(c)).convert('RGB'),np.float32)
 except Exception: return None
 h=(a[:,:,0]*65536+a[:,:,1]*256+a[:,:,2])*.01
 h[((a[:,:,0]==128)&(a[:,:,1]==0)&(a[:,:,2]==0))|((a[:,:,0]==0)&(a[:,:,1]==0)&(a[:,:,2]==0))]=np.nan
 return h.astype(np.float32)

def tile(z,x,y,mode,pref):
 urls=[]
 if mode=='coarse':
  urls=[(f'https://cyberjapandata.gsi.go.jp/xyz/dem5a_png/{z}/{x}/{y}.png','g','.png'),(f'https://cyberjapandata.gsi.go.jp/xyz/dem5b_png/{z}/{x}/{y}.png','g','.png'),(f'https://cyberjapandata.gsi.go.jp/xyz/dem5c_png/{z}/{x}/{y}.png','g','.png')]
 elif mode=='fine':
  if pref=='愛知': urls += [(f'https://rinya-tiles.geospatial.jp/dem_079_2025/{z}/{x}/{y}.png','g','.png'),(f'https://rinya-tiles.geospatial.jp/dem_078_2025/{z}/{x}/{y}.png','g','.png')]
  if pref=='三重': urls += [(f'https://rinya-tiles.geospatial.jp/dem_081_2025/{z}/{x}/{y}.png','g','.png')]
 elif mode=='fine1':
  urls=[(f'https://cyberjapandata.gsi.go.jp/xyz/dem1a_png/{z}/{x}/{y}.png','g','.png'),(f'https://qchizu3.xsrv.jp/mapdata/d52001/{z}/{x}/{y}.webp','q','.webp'),(f'https://gbank.gsj.jp/seamless/elev2/gsidem1a/{z}/{x}/{y}.webp','g','.webp')]
 for u,k,sfx in urls:
  c=get(u,sfx)
  if c:
   a=dec_q(c) if k=='q' else dec_gsi(c)
   if a is not None and np.isfinite(a).any(): return a
 return None

def mosaic(b,z,mode,pref):
 x0f,y1f=ll2t(b[0],b[1],z); x1f,y0f=ll2t(b[2],b[3],z)
 x0,x1=math.floor(x0f),math.floor(x1f); y0,y1=math.floor(y0f),math.floor(y1f)
 a=np.full(((y1-y0+1)*256,(x1-x0+1)*256),np.nan,np.float32); n=0
 for y in range(y0,y1+1):
  for x in range(x0,x1+1):
   q=tile(z,x,y,mode,pref)
   if q is not None: a[(y-y0)*256:(y-y0+1)*256,(x-x0)*256:(x-x0+1)*256]=q; n+=1
 if not n:return None,None
 meta={'z':z,'x0':x0,'y0':y0}
 px0,py1=ll2t(b[0],b[1],z); px1,py0=ll2t(b[2],b[3],z)
 X0=max(0,int((px0-x0)*256)); X1=min(a.shape[1],int(math.ceil((px1-x0)*256)))
 Y0=max(0,int((py0-y0)*256)); Y1=min(a.shape[0],int(math.ceil((py1-y0)*256)))
 a=a[Y0:Y1,X0:X1]; meta['x0']=x0+X0/256; meta['y0']=y0+Y0/256
 return a,meta

def p2ll(x,y,m): return t2ll(m['x0']+x/256,m['y0']+y/256,m['z'])
def ll2p(lon,lat,m):
 x,y=ll2t(lon,lat,m['z']); return (x-m['x0'])*256,(y-m['y0'])*256

def fill(a):
 ok=np.isfinite(a)
 if ok.all() or ok.sum()<100:return a
 ind=ndimage.distance_transform_edt(~ok,return_distances=False,return_indices=True); b=a.copy(); b[~ok]=a[tuple(ind[:,~ok])]; return b

def metrics(a,lat,z):
 b=fill(a); d=mpp(lat,z); gy,gx=np.gradient(b,d,d); sl=np.degrees(np.arctan(np.hypot(gx,gy)))
 rel=ndimage.maximum_filter(b,11)-ndimage.minimum_filter(b,11)
 return sl,rel

def components(a,m,b,maxn=7):
 sl,rel=metrics(a,(b[1]+b[3])/2,m['z']); ok=np.isfinite(a)
 wall=(sl>=44)&(rel>=7)&ok; bench=ndimage.binary_opening(sl<=18,np.ones((3,3))); cand=wall&ndimage.binary_dilation(bench,iterations=4)
 cand=ndimage.binary_opening(ndimage.binary_closing(cand,np.ones((3,3))),np.ones((2,2)))
 lab,n=ndimage.label(cand); pxm=mpp((b[1]+b[3])/2,m['z']); out=[]
 for k in range(1,n+1):
  ys,xs=np.where(lab==k)
  if len(xs)<4:continue
  area=len(xs)*pxm*pxm
  if area<18:continue
  s=sl[ys,xs]; r=rel[ys,xs]; j=int(np.nanargmax(s+r))
  lon,lat=p2ll(float(xs[j]),float(ys[j]),m)
  if b[0]<=lon<=b[2] and b[1]<=lat<=b[3]:out.append({'lon':lon,'lat':lat,'coarse':float(np.percentile(s,90)*1.5+np.percentile(r,90)*1.3+math.log1p(area)*4)})
 out.sort(key=lambda x:x['coarse'],reverse=True); sel=[]
 for q in out:
  if all(math.hypot((q['lon']-s['lon'])*111320*math.cos(math.radians(q['lat'])),(q['lat']-s['lat'])*110540)>450 for s in sel):sel.append(q)
  if len(sel)>=maxn:break
 return sel

def box(lon,lat,h=260):
 return lon-h/(111320*math.cos(math.radians(lat))),lat-h/110540,lon+h/(111320*math.cos(math.radians(lat))),lat+h/110540

def validate(lon,lat,pref):
 if pref in ('愛知','三重'):
  a,m=mosaic(box(lon,lat),18,'fine',pref)
  src='林野庁0.5m DEM'
  if a is None or np.isfinite(a).mean()<.35:a,m=None,None
 else:a,m=None,None
 if a is None:
  a,m=mosaic(box(lon,lat),17,'fine1',pref); src='DEM1A/Q地図 1m'
 if a is None or np.isfinite(a).mean()<.35:return None,None,None
 sl,rel=metrics(a,lat,m['z']); ok=np.isfinite(a); wall=(sl>=56)&(rel>=4)&ok
 lab,n=ndimage.label(ndimage.binary_opening(wall,np.ones((2,2)))); pxm=mpp(lat,m['z']); px0,py0=ll2p(lon,lat,m); best=None
 for k in range(1,n+1):
  ys,xs=np.where(lab==k)
  if len(xs)<8:continue
  cx,cy=float(xs.mean()),float(ys.mean()); dist=math.hypot((cx-px0)*pxm,(cy-py0)*pxm)
  if dist>190:continue
  comp=np.zeros(a.shape,bool); comp[ys,xs]=1; z15=ndimage.binary_dilation(comp,iterations=max(3,int(15/pxm)))
  bench=((sl<=20)&z15); bench_area=float(bench.sum()*pxm*pxm)
  z25=ndimage.binary_dilation(comp,iterations=max(4,int(25/pxm))); vals=a[z25&ok]
  h=float(np.percentile(vals,95)-np.percentile(vals,5)) if len(vals)>20 else 0; ms=float(np.percentile(sl[ys,xs],95))
  if h<5 or bench_area<8 or ms<56:continue
  j=int(np.argmax(sl[ys,xs]+rel[ys,xs])); LON,LAT=p2ll(float(xs[j]),float(ys[j]),m)
  score=ms*1.5+h*3.1+min(bench_area,600)*.035+math.log1p(len(xs)*pxm*pxm)*3-dist*.04
  q={'lon':LON,'lat':LAT,'fine':score,'wall_height_proxy_m':h,'max_slope':ms,'bench_area_m2':bench_area,'wall_area_m2':len(xs)*pxm*pxm}
  if best is None or q['fine']>best['fine']:best=q
 return best,a,m if best else (None,None,None)

def image_tile(lon,lat,z,layer,ext):
 xf,yf=ll2t(lon,lat,z); xc,yc=int(xf),int(yf); h=2; can=Image.new('RGB',(1280,1280),(210,210,210))
 for j,y in enumerate(range(yc-h,yc+h+1)):
  for i,x in enumerate(range(xc-h,xc+h+1)):
   c=get(f'https://cyberjapandata.gsi.go.jp/xyz/{layer}/{z}/{x}/{y}.{ext}','.'+ext)
   if c:
    try:can.paste(Image.open(io.BytesIO(c)).convert('RGB'),(i*256,j*256))
    except:pass
 px=(xf-(xc-h))*256;py=(yf-(yc-h))*256;d=ImageDraw.Draw(can);d.ellipse((px-12,py-12,px+12,py+12),outline='red',width=4);d.line((px-22,py,px+22,py),fill='red',width=3);d.line((px,py-22,px,py+22),fill='red',width=3)
 return can

def demimgs(a,m,lon,lat):
 b=fill(a);gy,gx=np.gradient(b);az=math.radians(315);al=math.radians(42);sr=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gx,gy);q=np.sin(al)*np.cos(sr)+np.cos(al)*np.sin(sr)*np.cos(az-asp);lo,hi=np.percentile(q,[1,99]);q=((q-lo)/(hi-lo)*255).clip(0,255);hill=Image.fromarray(q.astype('uint8')).convert('RGB');sl,_=metrics(a,lat,m['z']);sli=Image.fromarray((sl.clip(0,80)/80*255).astype('uint8')).convert('RGB');px,py=ll2p(lon,lat,m)
 for im in (hill,sli):d=ImageDraw.Draw(im);d.ellipse((px-10,py-10,px+10,py+10),outline='red',width=4);d.line((px-20,py,px+20,py),fill='red',width=3);d.line((px,py-20,px,py+20),fill='red',width=3)
 return hill,sli

def panel(r,a,m,i):
 ims=list(demimgs(a,m,r['lon'],r['lat']))+[image_tile(r['lon'],r['lat'],18,'seamlessphoto','jpg'),image_tile(r['lon'],r['lat'],17,'ort_USA10','png')];ims=[x.resize((900,900)) for x in ims];can=Image.new('RGB',(1800,1850),(18,22,29));d=ImageDraw.Draw(can)
 for k,im in enumerate(ims):can.paste(im,((k%2)*900,50+(k//2)*900))
 d.text((12,14),f"{i:02d} {r['region_name']}  {r['lat']:.6f}, {r['lon']:.6f}",fill='white');p=OUT/f'point_{i:02d}.jpg';can.save(p,quality=94);return p.name

def main():
 coarse=[];status=[]
 for rid,pref,name,b in REGIONS:
  print('coarse',rid,flush=True);a,m=mosaic(b,15,'coarse',pref)
  if a is None:status.append({'region':rid,'status':'no coarse dem'});continue
  cs=components(a,m,b);status.append({'region':rid,'status':'ok','coarse_candidates':len(cs)})
  for q in cs:q.update({'region_id':rid,'pref':pref,'region_name':name})
  coarse+=cs
 pd.DataFrame(status).to_csv(OUT/'region_status.csv',index=False);pd.DataFrame(coarse).to_csv(OUT/'coarse_candidates.csv',index=False)
 val=[]
 for q in sorted(coarse,key=lambda x:x['coarse'],reverse=True)[:50]:
  print('fine',q['region_id'],flush=True);v,a,m=validate(q['lon'],q['lat'],q['pref'])
  if v:v.update(q);v['_a']=a;v['_m']=m;v['total']=v['fine']+v['coarse']*.25;val.append(v)
 val.sort(key=lambda x:x['total'],reverse=True);final=[]
 for q in val:
  if all(math.hypot((q['lon']-s['lon'])*111320*math.cos(math.radians(q['lat'])),(q['lat']-s['lat'])*110540)>500 for s in final):final.append(q)
  if len(final)>=12:break
 rows=[]
 for i,q in enumerate(final,1):
  a=q.pop('_a');m=q.pop('_m');q['rank']=i;q['image']=panel(q,a,m,i);q['gsi_url']=f"https://maps.gsi.go.jp/#18/{q['lat']:.6f}/{q['lon']:.6f}/&base=seamlessphoto&ls=seamlessphoto&disp=1";q['photo_search_url']=f"https://service.gsi.go.jp/map-photos/app/map?color_type_ids=1%2C2&lat_max={q['lat']+.0015:.7f}&lat_min={q['lat']-.0015:.7f}&lon_max={q['lon']+.0018:.7f}&lon_min={q['lon']-.0018:.7f}&scale_from=0&scale_to=99999999&search=photo&search_date_from=0000&search_date_to=9999";rows.append(q)
 pd.DataFrame(rows).to_csv(OUT/'final_points.csv',index=False)
 geo={'type':'FeatureCollection','features':[{'type':'Feature','geometry':{'type':'Point','coordinates':[r['lon'],r['lat']]},'properties':{k:v for k,v in r.items() if k not in ('lon','lat')}} for r in rows]};(OUT/'final_points.geojson').write_text(json.dumps(geo,ensure_ascii=False,indent=2))
 cards=''.join([f"<article><h2>{r['rank']}. {r['region_name']}</h2><p><b>{r['lat']:.6f}, {r['lon']:.6f}</b>　壁高差推定 {r['wall_height_proxy_m']:.1f}m / 傾斜 {r['max_slope']:.1f}° / 棚 {r['bench_area_m2']:.0f}m²</p><img src='{r['image']}'><p><a href='{r['gsi_url']}'>航空写真</a>　<a href='{r['photo_search_url']}'>原版単写真検索</a></p></article>" for r in rows])
 (OUT/'report.html').write_text("<!doctype html><meta charset='utf-8'><style>body{background:#10151d;color:#eef;font-family:sans-serif;max-width:1500px;margin:auto;padding:25px}article{background:#1a222e;padding:20px;margin:22px 0;border-radius:12px}img{width:100%}a{color:#5bdcff}</style><h1>東海3県 DEM形状確認済み候補点</h1><p>遺跡認定ではなく、急壁＋直下棚の地形条件を通過した点です。</p>"+cards,encoding='utf-8')
 shutil.make_archive(str(ROOT/'tokai_scan_results'),'zip',OUT);print('DONE',len(rows),flush=True)

if __name__=='__main__':main()
