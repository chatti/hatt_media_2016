function [model] = CreateModelGPU_DRR(N)

info = mha_read_header(['data/segmentation.mhd']);
seg = mha_read_volume(info);
%lbls = seg;
seg = seg>0;


probematfile = ['data/probeDRR.mat'];
if ~exist(probematfile,'file')
    info = mha_read_header(['data/probe.mhd']);
    probe = mha_read_volume(info);
    probe = single(probe);
    probe = probe/max(probe(:));
    mv = mean(probe(find(seg==0)));
    probe = probe-mv;
    probe(probe<0)=0;
    probe = probe/max(probe(:));
     
else
    load(probematfile);
end


pxdim = info.PixelDimensions;
sz    = size(probe);


segdilate = imdilate(seg>0,ones(5,5,5));
probe = probe.*segdilate;
%probe(seg==1) = mean(probe(seg==1));
%probe(seg==2) = max(probe(seg==2));
%probe(seg==3) = mean(probe(seg==3));
%probe(seg==4) = mean(probe(seg==4));

if exist(probematfile,'file')
    save(probematfile,'probe', 'allidx','info');
end

cubesize=4;
if ~exist(probematfile,'file')
    allidx=zeros(prod(ceil(sz/cubesize)),cubesize^3);
    cnt=1;
    t=1;
    for i=1:cubesize:sz(1)
        100*(cnt/(ceil((sz(1)/cubesize))))
        r = i:(i+cubesize-1);
        for j=1:cubesize:sz(2)
            c = j:(j+cubesize-1);
            for k=1:cubesize:sz(3)
                z = k:(k+cubesize-1);
                r2 = [r(1)*ones(16,1);r(2)*ones(16,1);r(3)*ones(16,1);r(4)*ones(16,1)];
                a = [c(1)*ones(4,1); c(2)*ones(4,1); c(3)*ones(4,1); c(4)*ones(4,1)];
                c2 = [a;a;a;a];
                z=z';
                z2 = repmat(z,[16,1]);
                
                %tmp(r,c,z) = 1;
                %idx = find(tmp==1);
                %tmp(r,c,z) = 0;
                
                try
                    idx = sub2ind(sz,r2,c2,z2);
                catch
                    idx = [];
                end
                if isempty(idx)
                    
                elseif any(seg(idx)>0)
                    allidx(t,1:length(idx)) = idx;
                end
                t=t+1; 
            end
        end
        cnt=cnt+1;
    end

    rmidx = any(allidx==0,2);
    allidx(rmidx,:) = [];
    probe = probe./max(probe(:));
    save(probematfile,'probe','allidx','info');
else
    load(probematfile);
end


h = fspecial3('gaussian',[7,7,7]);
fprobe = imfilter(double(seg>0),h,'same');
fprobe = imfilter(fprobe,h,'same');
fprobe = imfilter(fprobe,h,'same');
[gx,gy,gz] = gradient(fprobe);
gx = gx.*(seg>0);
gy = gy.*(seg>0);
gz = gz.*(seg>0);
gmag = sqrt(gx.^2+gy.^2+gz.^2);
mxgmag = max(gmag(:));
gx=gx./gmag;
gy=gy./gmag;
gz=gz./gmag;
gx(gmag<(0.75*mxgmag)) = 0;
gy(gmag<(0.75*mxgmag)) = 0;
gz(gmag<(0.75*mxgmag)) = 0;


allbins = zeros(size(allidx));
for k=1:size(allbins,1)
    allbins(k,:) = k;
end
tmp = ones(sz);
tmp(allidx(:))=0;
bg = mean(probe(tmp==1));
probe = probe-bg;
probe(probe<0) = 0;
probe = probe/max(probe(:));
[y,x,z] = ind2sub(sz,allidx(:));
x=x + 1*(rand(length(allidx(:)),1)-1);
y=y + 1*(rand(length(allidx(:)),1)-1);
z=z + 1*(rand(length(allidx(:)),1)-1);
v   = interpn(probe,y,x,z);

i = rand(length(v),1);
[~,k] = sort(i);
k=k(1:N);

allbins = allbins(k);
[~,skey] = sort(allbins);
k=k(skey);

pnts = [pxdim(1)*x(k),pxdim(2)*y(k),pxdim(3)*z(k)];
pnts(:,1) = pnts(:,1)-8;
pnts(:,2) = pnts(:,2)-12;
pnts(:,3) = pnts(:,3)-8;

v(isnan(v)) = 0;
model.pnts = pnts;
model.vals = v(k);


px=0.25;
f=100000;
w=512;
h=512;

x = pnts(:,1);
y = pnts(:,2);
z = pnts(:,3);

x = x.*(f./(f-z))*(1/px);
y = y.*(f./(f-z))*(1/px);

x = x+((w-1)/2);
y = y+((h-1)/2);

x = w*(x/w);
y = h*(1-(y/h));

rings = CreateRings();
rings(:,1) = rings(:,1) + 76*pxdim(1)-8;
rings(:,2) = rings(:,2) + 300*pxdim(2)+2;
rings(:,3) = rings(:,3) + 62*pxdim(3)-8;

ringvals = 0.10*ones(size(rings,1),1);
model.pnts = [model.pnts;rings];
model.vals = [model.vals;ringvals];

p = model.pnts;
r = rings;
figure(1)
subplot(1,3,1)
hold off;
plot(p(1:100:end,1),p(1:100:end,2),'g.');xlabel('x');ylabel('y');axis equal;grid on;hold on
plot(r(1:100:end,1),r(1:100:end,2),'b.');xlabel('x');ylabel('y');axis equal;grid on;hold off

subplot(1,3,2)
hold off;
plot(p(1:100:end,1),p(1:100:end,3),'g.');xlabel('x');zlabel('z');axis equal;grid on;hold on
plot(r(1:100:end,1),r(1:100:end,3),'b.');xlabel('x');zlabel('z');axis equal;grid on;hold off

subplot(1,3,3)
hold off;
plot(p(1:100:end,2),p(1:100:end,3),'g.');ylabel('y');zlabel('z');axis equal;grid on;hold on
plot(r(1:100:end,2),r(1:100:end,3),'b.');ylabel('y');zlabel('z');axis equal;grid on;hold off

figure(2)
p = 8*model.pnts(:,[1,2]);
p=p+512/2;
p=round(p);
oob = p(:,1) < 1 | p(:,2) < 1 | p(:,2) > 512 | p(:,2) > 512;
p(oob,:) = [];
vx = model.vals;
vx(oob) = [];
img = accumarray(fliplr(round(p(:,1:2))),vx,[w,h]);
%Gx = accumarray(fliplr(round(p(:,1:2))),nx,[w,h]);
%Gy = accumarray(fliplr(round(p(:,1:2))),ny,[w,h]);
%subplot(1,3,1)
imshow(sqrt(img),[])
%subplot(1,3,2)
%imshow((Gx),[])
%subplot(1,3,3)
%imshow((Gy),[])

% colors = {'r.','g.','y.','b.','c.','m.'};
% for k=1:(1000*32):size(p,1)-32
%     r = ceil(5*rand(1,1));
%     hold on; plot(p(k:k+32,1),p(k:k+32,2),colors{r});hold off;
% end

save(probematfile,'probe','allidx','model','img');

ct.data = probe;
ct.PixelDimensions = info.PixelDimensions;
ct.CenterOfRotation = [8,12,8];
save ct.mat ct;
fid = fopen('ct.raw','w');
fwrite(fid,single(permute(probe,[2,1,3])),'single');
fclose(fid);

wtf=0;


function rings = CreateRings()

N=50;

cnt=30000;
r = 4.25;
s = 1;
t = linspace(0,2*pi,cnt);
x = r*cos(t);x=x';
z = r*sin(t);z=z';
y = zeros(cnt,1);
p = [x,y,z];
spread = s*(rand(cnt,3)-0.5);
p=p+spread;

rings=[];
hold off;
for k=1:N;
    g=[];
    for n=1:3
        t = p;
        t(:,2) = t(:,2) + ((n-1)*s*0.5);
        g = [g;t];
    end
    tmp = g;
    tmp(:,2) = tmp(:,2) + ((k-1)*s*2.5);
    rings = [rings;tmp];
end
 
