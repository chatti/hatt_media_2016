function [model] = CreateModelGPU(N)

info = mha_read_header(['data/segmentation.mhd']);
seg = mha_read_volume(info);

%info = mha_read_header([DropboxRoot,'/TEE_XRF_Registration/SimpleModel.mhd']);
%simplemodel = mha_read_volume(info);

lbls = seg;

[ip(:,2),ip(:,1),ip(:,3)] = ind2sub(size(seg),find(seg>7));

seg(seg==5)  = 3;
seg(seg==6)  = 3;
seg(seg==7)  = 3;

seg(seg==8)  = 2;
seg(seg==9)  = 4;
%seg(seg==10) = 0;
seg(seg==11) = 5;
seg(seg==12) = 5;
seg(seg==13) = 5;

probematfile = [pwd,filesep,'probe.mat'];

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
probe(seg==2) = mean(probe(seg==2));
%probe(seg==3) = mean(probe(seg==3));
%probe(seg==4) = mean(probe(seg==4));

if exist(probematfile,'file')
    save probe.mat probe allidx;
end

cubesize=4;
if ~exist(probematfile,'file')
    allidx=zeros(prod(ceil(sz/cubesize)),cubesize^3);
    tmp = zeros(sz);
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
    save(probematfile,'probe','allidx');
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
gx(gmag==0) = 0;
gy(gmag==0) = 0;
gz(gmag==0) = 0;


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
lbl = interpn(double(lbls),y,x,z,'nearest');
nx = interpn(gx,y,x,z,'linear');
ny = interpn(gy,y,x,z,'linear');
nz = interpn(gz,y,x,z,'linear');
i = rand(length(v),1);
[~,k] = sort(i);
k=k(1:N);

allbins = allbins(k);
[~,skey] = sort(allbins);
k=k(skey);

%simpleseg = simplemodel>0;
%simpleseg = bwulterode(simpleseg);
%[simy,simx,simz] = ind2sub(sz,find(simpleseg==1));

%spnts = [pxdim(1)*simx,pxdim(2)*simy,pxdim(3)*simz];
%spnts(:,1) = spnts(:,1)-8;
%spnts(:,2) = spnts(:,2)-12;
%spnts(:,3) = spnts(:,3)-8;

pnts = [pxdim(1)*x(k),pxdim(2)*y(k),pxdim(3)*z(k)];
pnts(:,1) = pnts(:,1)-8;
pnts(:,2) = pnts(:,2)-12;
pnts(:,3) = pnts(:,3)-8;

ip = [pxdim(1)*ip(:,1),pxdim(2)*ip(:,2),pxdim(3)*ip(:,3)];
ip(:,1) = ip(:,1)-8;
ip(:,2) = ip(:,2)-12;
ip(:,3) = ip(:,3)-8;

model.keypoints = ...
    [ 0,12  ,-1;
     -5,4 ,4.5;
     -5,-5  ,4.5;
     5 ,4 ,4.5;
     5 ,-5  ,4.5;
     -3,-10  ,0;
     3 ,-10  ,0;
     0 ,-3.5,-2.5;
     -5,   8,0;
     5 ,   8,0;
     -6,   0,0;
     6 ,   0,0;
     -4,  15,0;
      4,  15,0;
      -4, 24,-1;
      4, 24,-1;
      0, 24,2;
      0, 24,-6;
      0, 15,-5;
      0,  6,-5;
      0,  2,1;
      0, -2,1;
     -5, -7,0;
      5, -7,0;];

kp = model.keypoints;
model.keypointsCT      = (1/pxdim(1))*(model.keypoints(:,1) + 8);
model.keypointsCT(:,2) = (1/pxdim(2))*(model.keypoints(:,2) + 12);
model.keypointsCT(:,3) = (1/pxdim(3))*(model.keypoints(:,3) + 8);

%model.simpleModel = spnts;

v(isnan(v)) = 0;
model.pnts = pnts;
model.vals = v(k);
model.part = lbl(k);
nx=nx(k);
ny=ny(k);
nz=nz(k);
model.G = [nx,ny,nz];

kp([17:20,23,24],:) = [];

colors = {'r.','g.','y.','b.','c.','m.'};
figure(3)
% for k=1:16:(size(pnts,1)-16)
%     rndidx = ceil(6*rand(1));
%     plot3(pnts(k:(k+16),1),pnts(k:(k+16),2),pnts(k:(k+16),3),[colors{rndidx}],'MarkerSize',7);
%     hold on;
% end
% hold off;

C = 1:size(pnts,1);
scatter3(pnts(:,1),pnts(:,2),pnts(:,3),1,C);
xlabel('x');
ylabel('y');
zlabel('z');
set(gca,'color','k')
axis equal;
hold off;

figure(1)
subplot(1,2,2);
plot3(kp(:,1),kp(:,2),kp(:,3),'go','MarkerFaceColor','g');
hold on;
for k=1:size(kp,1)
   text(kp(k,1),kp(k,2),kp(k,3),num2str(k),'Fontsize',18,'Color','r');
end
xlabel('x');
ylabel('y');
zlabel('z');
axis equal;
hold off;
set(gca,'color','w');
set(gcf,'color','w');


px=0.30;
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

ix = kp(:,1);
iy = kp(:,2);
iz = kp(:,3);

ix = ix.*(f./(f-iz))*(1/px);
iy = iy.*(f./(f-iz))*(1/px);

ix = ix+((w-1)/2);
iy = iy+((h-1)/2);

ix = w*(ix/w);
iy = h*(1-(iy/h));

p = [x,y]; 
img = accumarray(fliplr(round(p(:,1:2))),model.vals,[w,h]);
h = fspecial('gaussian',[11,11],0.5);
img = imfilter(img,h,'same');
%Gx = accumarray(fliplr(round(p(:,1:2))),nx,[w,h]);
%Gy = accumarray(fliplr(round(p(:,1:2))),ny,[w,h]);
%subplot(1,3,1)
figure(2)
imshow(-sqrt(img),[]);
hold on;
plot(ix,iy,'.r');
for k=1:length(ix)
   text(ix(k),iy(k),num2str(k),'Fontsize',18,'Color','r');
   pos = [ix(k)-8,iy(k)-8,16,16];
   rectangle('Position',pos,'EdgeColor','r')
end
hold off;
set(gca,'color','w');
set(gcf,'color','w');




