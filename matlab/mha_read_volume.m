function V = mha_read_volume(info)
% Function for reading the volume of a Insight Meta-Image (.mha, .mhd) file
% 
% volume = tk_read_volume(file-header)
%
% examples:
% 1: info = mha_read_header()
%    V = mha_read_volume(info);
%    imshow(squeeze(V(:,:,round(end/2))),[]);
%
% 2: V = mha_read_volume('test.mha');

if(~isstruct(info)), info=mha_read_header(info); end


switch(lower(info.DataFile))
    case 'local'
    otherwise
    % Seperate file
    info.Filename=fullfile(fileparts(info.Filename),info.DataFile);
end
        
% Open file
if isfield(info,'byteorder')
    switch(info.ByteOrder(1))
        case ('true')
            fid=fopen(info.Filename','rb','ieee-be');
        otherwise
            fid=fopen(info.Filename','rb','ieee-le');
    end
else
    fid=fopen(info.Filename','rb','ieee-le');
end

switch(lower(info.DataFile))
    case 'local'
        % Skip header
        fseek(fid,info.HeaderSize,'bof');
    otherwise
        fseek(fid,0,'bof');
end

if isfield(info,'ElementNumberOfChannels')
    datasize=prod(info.Dimensions)*info.ElementNumberOfChannels*info.BitDepth/8;
else
    datasize=prod(info.Dimensions)*info.BitDepth/8;    
end

switch(info.CompressedData(1))
    case 'f'
        % Read the Data
        switch(info.DataType)
            case 'char'
                 V = fread(fid,datasize,'*int8'); 
            case 'uchar'
                V = fread(fid,datasize,'*uint8'); 
            case 'short'
                V = fread(fid,datasize,'*int16'); 
            case 'ushort'
                V = fread(fid,datasize,'*uint16'); 
            case 'int'
                 V = fread(fid,datasize,'*int32'); 
            case 'uint'
                 V = fread(fid,datasize,'*uint32'); 
            case 'float'
                 V =fread(fid,datasize,'*single');   
            case 'double'
                V = fread(fid,datasize,'*double');
        end
    case 't'
        switch(info.DataType)
            case 'char', DataType='int8';
            case 'uchar', DataType='uint8';
            case 'short', DataType='int16';
            case 'ushort', DataType='uint16';
            case 'int', DataType='int32';
            case 'uint', DataType='uint32';
            case 'float', DataType='single';
            case 'double', DataType='double';
        end
        Z  = fread(fid,inf,'uchar=>uint8');
        V = zlib_decompress(Z,DataType);

end
fclose(fid);
if isfield(info,'ElementNumberOfChannels')
    V = reshape(V,[info.Dimensions,info.ElementNumberOfChannels]);
else
    V = reshape(V,info.Dimensions);  
end


function M = zlib_decompress(Z,DataType)
import com.mathworks.mlwidgets.io.InterruptibleStreamCopier
a=java.io.ByteArrayInputStream(Z);
b=java.util.zip.InflaterInputStream(a);
isc = InterruptibleStreamCopier.getInterruptibleStreamCopier;
c = java.io.ByteArrayOutputStream;
isc.copyStream(b,c);
M=typecast(c.toByteArray,DataType);

