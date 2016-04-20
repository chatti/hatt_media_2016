function [] = XUS_Simulation(doplot,doexport)
%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%
%Set parameters for simulations here
pointcount = 2^16;  %Size of the point cloud
xraycount = 1;
metrics = [1,2,3,4,5];%1-DSC, 2-PatchGCC, 3-rcNCC, 4-rcGCC, 5-DSC-PatchGCC-Hybrid
optimizer = 1;%1-Nelder-mead, 2-Powell, 3-LBGFS
%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%

load data/pcloud.model.20.mat;
load data/ct.mat;

dispmodel=model;

%Setup space of possible object 
%out-of-plane pose parameters
pitch= deg2rad(-30:15:30);
roll = deg2rad(-0:15:90);
    
%Get a random index for the roll and pitch of the probe
pitchIdx = ceil(length(pitch)*rand(1));
rollIdx  = ceil(length(roll) *rand(1));

%Get a random in-plane rotation and translation
yaw = deg2rad(45*(rand(1)-0.5));
tx  = (10*(rand(1)-0.5));
ty  = (-10*(rand(1)));

%Get a random x-ray detector pitch from 0.25 to 0.35 mm
xrayPitch = 0.1*rand(1)+0.25;
%Fix SID at 1200
sid = 1200;

%Get a random isocenter ranging from 1/4 to 1/3 of SID
isocenter = (0.3333-.25)*rand(1) +0.25;
isocenter = isocenter*sid;
%Setup the XRF camera parameters
camera1.w=512;
camera1.h=512;
camera1.px=xrayPitch;
camera1.py=xrayPitch;

camera1.sid=sid;
camera1.isocenter=isocenter;

rx = pitch(pitchIdx);
ry = roll(rollIdx);

%The initial transform defines the true object pose
initialTransform   = [tx, ty, camera1.isocenter, rx,ry, yaw];

%The secondary transform defines the misaligment
%1-tx, 2-ty, 3-tz, 4-rx, 5-ry, 6-rz
secondaryTransform = [0,0,0,0,0,0];
secondaryTransform(1) = 5.0*(rand(1)-0.5);
secondaryTransform(2) = 5.0*(rand(1)-0.5);
secondaryTransform(3) = 5.0*(rand(1)-0.5);
secondaryTransform(4) = (pi/4)*(rand(1)-0.5);
secondaryTransform(5) = (pi/4)*(rand(1)-0.5);
secondaryTransform(6) = (pi/12)*(rand(1)-0.5);

%Don't rotate the camera
camera1.transform  = [0,0,0,0,0,0];

%Create a testinfo struct
testinfo.initialTransform       = initialTransform;
testinfo.secondaryTransform     = secondaryTransform;
testinfo.camera{1}              = camera1;

testinfo.numxrays               = xraycount;

testinfo.parameterfilename      = [pwd,'/../input/pfile.txt'];
testinfo.logfilename          = [pwd,'/../output/log.txt'];
testinfo.simxrayfilename     = [pwd,'/../input/simxray1.raw'];

%Create some randomly located fiducials within the simulated US
%volume
fiducials3D = 25*(rand(15,3)-0.5) + repmat([0,0,50],15,1);
testinfo.fiducials3D = fiducials3D;

sz = size(ct.data);
sz = sz([2,1,3]);
box = [0,0,0,1;
       sz(1),0,0,1;
       0,sz(2),0,1;
       sz(1),sz(2),0,1;
       0,0,sz(3),1;
       sz(1),0,sz(3),1;
       0,sz(2),sz(3),1;
       sz(1),sz(2),sz(3),1];
box(:,1) = box(:,1)*ct.PixelDimensions(1);
box(:,2) = box(:,2)*ct.PixelDimensions(2);
box(:,3) = box(:,3)*ct.PixelDimensions(3);

box(:,1) = box(:,1)-ct.CenterOfRotation(1);
box(:,2) = box(:,2)-ct.CenterOfRotation(2);
box(:,3) = box(:,3)-ct.CenterOfRotation(3);

testinfo.box = box;

testinfo.ctfilename = [pwd,'/../input/ct.raw'];

load data/drrModel;


k=1;
[drr,testinfo.fiducialsT3D,testinfo.fiducialsT2D,~,~,~,~] = Splat(testinfo.initialTransform, testinfo.secondaryTransform, testinfo.camera{k}, drrModel, fiducials3D,testinfo.box);
drr=sqrt(drr); %add some contrast
testinfo.simxray = 1-drr;
                
fid = fopen(testinfo.simxrayfilename,'w');
fwrite(fid,fliplr(1-testinfo.simxray'),'single');
fclose(fid);
        
pcstr = num2str( log2(pointcount) );
%load a matlab variable named "model" that contains the object
%point cloud
load(['data/pcloud.model.',pcstr,'.mat']); 
testinfo.model=model;
testinfo.pointcloudfilename     = [pwd, '/../input/pc.raw'];
fid                             = fopen(testinfo.pointcloudfilename,'w');
probe                           = model.pnts;
probe(:,4)                      = model.vals;
probe(:,5)                      = model.G(:,1);
probe(:,6)                      = model.G(:,2);
probe(:,7)                      = model.G(:,3);
fwrite(fid,single(probe(:)),'single');
fclose(fid);
    
for m=metrics
    
    testinfo.metric=m;
    testinfo.optimizer = optimizer;
    testinfo.pointcloudsize=pointcount;
    WriteParameterFile(testinfo);
    
    RunSimulationCUDA(testinfo);
    disp(['Metric: ', num2str(testinfo.metric),' Optimizer: ', num2str(testinfo.optimizer)]);
    [testinfo.history, testinfo.finalTransform, testinfo.Timing,testinfo.functionEvals] = ReadHistory(testinfo);
    [~,curFid3D,curFid2D] = Splat(testinfo.initialTransform, testinfo.finalTransform, testinfo.camera{1}, model, fiducials3D,testinfo.box);
    [testinfo.tre2d,testinfo.tre3d] = TransformToTRE(testinfo,curFid2D,curFid3D);
    [~,curFid3D,curFid2D] = Splat(testinfo.initialTransform, [0,0,0,0,0,0], testinfo.camera{1}, model, fiducials3D,testinfo.box);
    [testinfo.ipriortre2d,testinfo.itre3d] = TransformToTRE(testinfo,curFid2D,curFid3D);
    disp(['iTRE2D: ', num2str(testinfo.ipriortre2d),'TRE2D: ', num2str(testinfo.tre2d),' TRE3D : ', num2str(testinfo.tre3d), ' fevals : ', num2str(testinfo.functionEvals), ' ms : ', num2str(testinfo.Timing)]);
    disp('---------------');

    if doplot
        figure(1)
        PlaybackHistory(testinfo,dispmodel,doexport);
    end
end

function testinfo = WriteParameterFile(testinfo)

    if testinfo.metric == 1
        WriteParameterFileDash(testinfo);
    elseif testinfo.metric == 2
        WriteParameterFilePatchRaycast(testinfo);
    elseif testinfo.metric == 3 || testinfo.metric == 4
        WriteParameterFileRaycast(testinfo);
    elseif testinfo.metric == 5
        oldpfile = testinfo.parameterfilename;
        testinfo.parameterfilename=[oldpfile,'1'];
        WriteParameterFileDash(testinfo);
        testinfo.parameterfilename=[oldpfile,'2'];
        WriteParameterFilePatchRaycast(testinfo);
        testinfo.parameterfilename=oldpfile;
    end


function [] = WriteParameterFileDash(testinfo)
fid = fopen([testinfo.parameterfilename],'wt');
fprintf(fid,'numxrays\t%d\n',testinfo.numxrays);

camera = testinfo.camera{1};
str = num2str([camera.w camera.h camera.px camera.py camera.sid camera.isocenter camera.transform]);
fprintf(fid,'camera1\t%s\n',str); 

fprintf(fid,'xrayfilename1\t%s\n',testinfo.simxrayfilename);

fprintf(fid,'pointcloudsize\t%d\n',testinfo.pointcloudsize);
fprintf(fid,'pointcloudfilename\t%s\n',testinfo.pointcloudfilename);
fprintf(fid,'initialtransform\t%s\n',num2str(testinfo.initialTransform));
fclose(fid);

function [] = WriteParameterFileRaycast(testinfo)
fid = fopen([testinfo.parameterfilename],'wt');
fprintf(fid,'numxrays\t%d\n',testinfo.numxrays);

camera = testinfo.camera{1};
str = num2str([camera.w camera.h camera.px camera.py camera.sid camera.isocenter camera.transform]);
str = [testinfo.simxrayfilename,' ',str];
fprintf(fid,'xray1\t%s\n',str); 

fprintf(fid,['ct ',testinfo.ctfilename,' 150 432 146 0.1037 0.1037 0.1037 8 12 8\n']);
fprintf(fid,'initialtransform %s\n',num2str(testinfo.initialTransform));
fprintf(fid,'raydelta 0.50\n');
fclose(fid);

function [] = WriteParameterFilePatchRaycast(testinfo)
fid = fopen([testinfo.parameterfilename],'wt');
fprintf(fid,'numxrays\t%d\n',testinfo.numxrays);

camera = testinfo.camera{1};
str = num2str([camera.w camera.h camera.px camera.py camera.sid camera.isocenter camera.transform]);
str = [testinfo.simxrayfilename,' ',str];
fprintf(fid,'xray1\t%s\n',str); 

fprintf(fid,['ct ',testinfo.ctfilename,' 150 432 146 0.1037 0.1037 0.1037 8 12 8\n']);
fprintf(fid,'initialtransform %s\n',num2str(testinfo.initialTransform));
fprintf(fid,'raydelta 0.50\n');
fprintf(fid,'keypoints \n');

testinfo.model.keypointsCT([17:20,23,24],:) = [];
testinfo.model.keypointsCT(:,4) = 1;
for k=1:length(testinfo.model.keypointsCT(:))
    fprintf(fid,num2str(testinfo.model.keypointsCT(k)));
    fprintf(fid,'\n');    
end
fclose(fid);


function [] = RunSimulationCUDA(testinfo)

if testinfo.metric==1
    str = [pwd,'/../cuda/bin/dsc ',    testinfo.parameterfilename, ' ', testinfo.logfilename,  ' 1 ' ,num2str(testinfo.optimizer)];
elseif testinfo.metric==2
    str = [pwd,'/../cuda/bin/patchraycast ', testinfo.parameterfilename, ' ', testinfo.logfilename, ' ' ,num2str(testinfo.optimizer), ' 2'];
elseif testinfo.metric==3
    str = [pwd,'/../cuda/bin/raycast ', testinfo.parameterfilename, ' ', testinfo.logfilename, ' ' ,num2str(testinfo.optimizer), ' 1'];
elseif testinfo.metric==4
    str = [pwd,'/../cuda/bin/raycast ', testinfo.parameterfilename, ' ', testinfo.logfilename, ' ' ,num2str(testinfo.optimizer), ' 2'];
elseif testinfo.metric==5
    str = [pwd,'/../cuda/bin/hybrid ', testinfo.parameterfilename, '1 ', testinfo.parameterfilename, '2 ', testinfo.logfilename, ' 1 ' ,num2str(testinfo.optimizer)];
end
[status,result] = system(str);

if (status~=0)
    error('Problem with the underlying code');
    result
end


function [tre2d,tre3d] = TransformToTRE(testinfo,curFid2D,curFid3D)
    trueFid2D = testinfo.fiducialsT2D;
    trueFid3D = testinfo.fiducialsT3D;
       
    fidDiff = (curFid2D - trueFid2D);
    mag = (testinfo.camera{1}.sid-mean(trueFid3D(:,3))) / testinfo.camera{1}.sid;
    fidDiff = testinfo.camera{1}.px*mag*fidDiff;
    tre2d = sum(fidDiff.^2,2);
    tre2d = sqrt(sum(tre2d)/size(tre2d,1));
    
    fidDiff = (curFid3D - trueFid3D);
    tre3d = sum(fidDiff.^2,2);
    tre3d = sqrt(sum(tre3d)/size(tre3d,1));

function [history, finalTransform, milliseconds,functionEvals] = ReadHistory(testinfo)

f = fopen(testinfo.logfilename);
history = fscanf(f,'%f');
milliseconds = history(end);
history(end-6:end) = [];
L = length(history);
history = reshape(history,[7,L/7])';
functionEvals = size(history,1);

if any(isnan(history(:)))
    testinfo.logfilename
    error('History has Nans');
end

finalTransform = history(functionEvals,1:6);
fclose(f);

function [] = PlaybackHistory(testinfo,model,doexport)


simxray = testinfo.simxray;
history = testinfo.history;
numxray = testinfo.numxrays;

if any(isnan(history(:)))
    error('History has Nans');
end

for k=1:size(history)

    for n=1:numxray
        img = simxray;
        [splatimg,curFid3D,curFid2D] = Splat(testinfo.initialTransform, history(k,1:6), testinfo.camera{n}, model, testinfo.fiducials3D,testinfo.box);
        splatimg = canny(splatimg,1.0);
        splatimg = phasesym(splatimg,1,2,3,1.3,.1,2.0,1,-1);
        splatimg = splatimg/max(splatimg(:));
        trueFid2D = testinfo.fiducialsT2D;
        colorimg = 0.75*img + 0.5*splatimg;
        colorimg(:,:,2) = 0.75*img + 0.5*splatimg;
        colorimg(:,:,3) = 0.75*img + 0.5*splatimg;
        %subplot(1,2,1)
        imshow(colorimg,[]);
        hold on;
        plot(curFid2D(:,1),curFid2D(:,2),'+w');
        plot(trueFid2D(:,1),trueFid2D(:,2),'ro','Markersize',10);
%        plot(simpleModelPnts(:,1),simpleModelPnts(:,2),'b.','MarkerSize',.5);
        tre = TransformToTRE(testinfo,curFid2D,curFid3D);
        xlabel(num2str(tre));
        methods = {'DSC', 'Patch Raycast GCC', 'Raycast NCC', 'Raycast GCC', 'Splat NCC', 'Splat GCC', 'Hybrid'};
        opt = {'Nelder-Mead', 'Powell'};
        
        txt{1} = ['Method: ', methods{testinfo.metric}];
        txt{2} = ['Optimizer: ', opt{testinfo.optimizer}];
        txt{3} = ['pTRE (mm): ', num2str(tre,'%.2f')];
        %txt{4} = ['Time passed (ms): ', num2str(k*testinfo.Timing/testinfo.functionEvals,'%.2f')];
        text(10,015,txt{1},'Color','w','FontSize',12,'EdgeColor','w','BackgroundColor','k');
        text(10,050,txt{2},'Color','w','FontSize',12,'EdgeColor','w','BackgroundColor','k');
        text(10,085,txt{3},'Color','w','FontSize',12,'EdgeColor','w','BackgroundColor','k');
        %text(10,110,txt{4},'Color','w','FontSize',18,'EdgeColor','w','BackgroundColor','k');
        text(10,120,'+ - Est. Target Location','Color','w','FontSize',12,'EdgeColor','w','BackgroundColor','k');
        text(10,155,'o - True Target Location','Color','r','FontSize',12,'EdgeColor','w','BackgroundColor','k');
        hold on;
        plot([20,512-20],[500,500],'-w')
        tlocation = 20 + (k*testinfo.Timing/testinfo.functionEvals)*((512-40)/500);
        plot(tlocation,500,'rv');
        text(tlocation,500-20,[num2str((k*testinfo.Timing/testinfo.functionEvals),'%.2f'),' ms'],'Color','w','FontSize',12,'EdgeColor','w','BackgroundColor','k');

        hold off;
        hold off
        
        hold off;
        set(gcf,'color','k')
        set(gcf, 'Position', [800, 100, 600, 600]);

        %subplot(1,2,2)
        %plot(1:size(history,1),-history(:,7),'-w')
        %set(gca,'color','k')
        %hold on; plot(k,-history(k,7),'ro');hold off;
        %axis square;
        pause(.005)
        o = num2str(testinfo.optimizer,'%d');
        m = num2str(testinfo.metric,'%d');
        figname = ['exports/fig.',num2str(k,'%.3d'),'.',m,'.',o,'.png'];
        %saveas(gcf,['exports/fig.',num2str(k,'%.3d'),'.',m,'.',o,'.pdf'])
        if doexport
            export_fig(figname);
        end
        pause(.005)
    end
end


function nimg = Extract(img,W,H)

[X,Y] = meshgrid(1:W,1:H);
nimg = zeros(H,W);
X = X(:) + size(img,2)/2 - W/2;
Y = Y(:) + size(img,1)/2 - H/2;
v = interpn(img,Y,X,'linear',0);
nimg(:) = v;

function mix = BlendImages2(bg,drr,a)

bg  = bg-min(bg(:));
bg  = bg/max(bg(:));

drr = drr-min(drr(:));
drr = drr/max(drr(:));

mix = bg.*exp(-a*drr);

mv = mean(mix(:));
mix(1,:)=mv;
mix(end,:)=mv;
mix(:,1)=mv;
mix(:,end)=mv;


function [F,C,R] = InitMatrices(init,secondary,cameratransform,isocenter)

iT = SetTranslation(init);
iR = SetRotation(init);
sT = SetTranslation(secondary);
sR = SetRotation(secondary);
cT = SetTranslation(cameratransform);
cR = SetRotation(cameratransform);

I = eye(4);
I(3,4) = isocenter;

R = sR*iR;
F = sT*iT*sR*iR;
C = I*inv(cR)*inv(cT)*inv(I);


function M = SetTranslation(transform)
M=eye(4);
M(1,4) = transform(1);
M(2,4) = transform(2);
M(3,4) = transform(3);


function M = SetRotation(transform)

M=eye(4);
cx = cos(transform(4));
sx = sin(transform(4));
cy = cos(-transform(5));
sy = sin(-transform(5));
cz = cos(transform(6));
sz = sin(transform(6));
M(1,1) = cy*cz+sx*sy*sz; M(1,2) = -cx*sz; M(1,3) = cy*sx*sz-cz*sy;
M(2,1) = cy*sz-cz*sx*sy; M(2,2) =  cx*cz; M(2,3) = -sy*sz-cy*cz*sx;
M(3,1) = cx*sy;          M(3,2) =  sx;    M(3,3) = cx*cy;


function [img,fiducials3D,fiducials2D,u,v,i,box2D] = Splat(init, secondary, camera, model, fiducials,box)

sid = camera.sid;
w   = camera.w;
h   = camera.h;
px  = camera.px;
py  = camera.py;

[F,C] = InitMatrices(init,secondary,camera.transform,camera.isocenter);

T = C*F;

x = model.pnts(:,1);
y = model.pnts(:,2);
z = model.pnts(:,3);
i = model.vals;

tx = fiducials(:,1);
ty = fiducials(:,2);
tz = fiducials(:,3);

dtx = tx*T(1,1) + ty*T(1,2) + tz*T(1,3)  + T(1,4);
dty = tx*T(2,1) + ty*T(2,2) + tz*T(2,3)  + T(2,4);
dtz = tx*T(3,1) + ty*T(3,2) + tz*T(3,3)  + T(3,4);


bx = box(:,1);
by = box(:,2);
bz = box(:,3);

btx = bx*T(1,1) + by*T(1,2) + bz*T(1,3)  + T(1,4);
bty = bx*T(2,1) + by*T(2,2) + bz*T(2,3)  + T(2,4);
btz = bx*T(3,1) + by*T(3,2) + bz*T(3,3)  + T(3,4);

fiducials3D = [dtx,dty,dtz];

box3D = [btx,bty,btz];

dx = x*T(1,1) + y*T(1,2) + z*T(1,3)  + T(1,4);
dy = x*T(2,1) + y*T(2,2) + z*T(2,3)  + T(2,4);
dz = x*T(3,1) + y*T(3,2) + z*T(3,3)  + T(3,4);

% figure(2)
% plot3(dx(1:100:end),dy(1:100:end),dz(1:100:end),'.');axis equal;
% xlabel('x');
% ylabel('y');
% zlabel('z');

f = sid./(sid-dz);
x = dx.*f.*(1/px);
y = dy.*f.*(1/py);

f2 = sid./(sid-dtz);
tx = dtx.*f2.*(1/px);
ty = dty.*f2.*(1/py);

f3 = sid./(sid-btz);
bx = btx.*f3.*(1/px);
by = bty.*f3.*(1/py);

x = x + w*0.5;
y = y + h*0.5;

tx = tx + w*0.5;
ty = ty + h*0.5;

bx = bx + w*0.5;
by = by + h*0.5;

fiducials2D = [tx,h-ty];
box2D = [bx,h-by];

u = round(x);
v = round(y);
oob = (u < 1) | (u > 512) | (v < 1) | (v > 512);
u(oob) = [];
v(oob) = [];
i(oob) = [];
idx = w*(v-1)+u;
img = accumarray(idx,i,[w*h,1]);

img = reshape(img,[w,h]);
img=img';
h = fspecial('gaussian',[7,7],0.50);
img=imfilter(img,h,'same');
%img=sqrt(img);
img=img/max(img(:));
img=flipud(img);

function [simpleModel2D,fiducials3D,fiducials2D] = SimpleSplat(init, secondary, camera, model, fiducials)

sid = camera.sid;
w   = camera.w;
h   = camera.h;
px  = camera.px;
py  = camera.py;

[F,C] = InitMatrices(init,secondary,camera.transform,camera.isocenter);

T = C*F;

tx = fiducials(:,1);
ty = fiducials(:,2);
tz = fiducials(:,3);

dtx = tx*T(1,1) + ty*T(1,2) + tz*T(1,3)  + T(1,4);
dty = tx*T(2,1) + ty*T(2,2) + tz*T(2,3)  + T(2,4);
dtz = tx*T(3,1) + ty*T(3,2) + tz*T(3,3)  + T(3,4);

bx = model.simpleModel(:,1);
by = model.simpleModel(:,2);
bz = model.simpleModel(:,3);

btx = bx*T(1,1) + by*T(1,2) + bz*T(1,3)  + T(1,4);
bty = bx*T(2,1) + by*T(2,2) + bz*T(2,3)  + T(2,4);
btz = bx*T(3,1) + by*T(3,2) + bz*T(3,3)  + T(3,4);

fiducials3D = [dtx,dty,dtz];

% figure(2)
% plot3(dx(1:100:end),dy(1:100:end),dz(1:100:end),'.');axis equal;
% xlabel('x');
% ylabel('y');
% zlabel('z');

f2 = sid./(sid-dtz);
tx = dtx.*f2.*(1/px);
ty = dty.*f2.*(1/py);

f3 = sid./(sid-btz);
bx = btx.*f3.*(1/px);
by = bty.*f3.*(1/py);

tx = tx + w*0.5;
ty = ty + h*0.5;

bx = bx + w*0.5;
by = by + h*0.5;

fiducials2D = [tx,h-ty];
simpleModel2D = [bx,h-by];


