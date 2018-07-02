%% My implementation of eigenfaces
clear;
close all;clc

load('YaleB_32x32');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Visualizing Original Faces\n\n');
X = double(fea); %X the input variable we will operate
                 % on
% figure(1),title('Original Images'); %displaying first 16 images.
% for i=1:16
%    subplot(4,4,i),imshow(reshape(uint8(fea(i,:)),32,32));
% end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X_avg = zeros(1,size(X,2));
for i=1:size(X,1),
    X_avg = X_avg + X(i,:);
end;
X_avg = X_avg/size(X,1);
X_norm = zeros(size(X,1),size(X,2));
for i=1:size(X,1),
    X_norm(i,:) = X(i,:) - X_avg;
end;

% figure(2);
% imshow(uint8(reshape(X_avg,32,32))),title('Average face');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[m n] = size(X_norm);
sigma = (1/m)*X_norm'*X_norm;
[U,S,V] = svd(sigma); % U matrix has the eigen vectors sorted in decending order.
% figure(3);
% for i=1:16
%    subplot(4,4,i),imshow(reshape(uint8(U(:,i))+X_avg,32,32));
% end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K =100;
Z = zeros(size(X_norm,1),K);
U_red = U(:,1:K);
Z = X_norm*U_red;
X_rec = Z*U_red';
% figure(4);
% for i=1:16,
%     subplot(4,4,i),imshow(reshape(uint8(X_rec(i,:))+X_avg,32,32));
% end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Recognition for a given input image----------
% In_img = imread('test1.jpeg'); %change this to change input image
% In_img = double(rgb2gray(In_img));
% Y = reshape(In_img,1,1024) - X_avg;
% figure(5);
% imshow(uint8(In_img));
% Y_in_eigenspace = Y*U_red;
% Y_rec = Y_in_eigenspace*U_red';
% epsilon = sum((Y - Y_rec).^2);
% figure(6);
% imshow(uint8(reshape(Y_rec + X_avg,32,32)));
% if (epsilon < 2.5e+05)
%     fprintf('Is a face\n');
% else
%     fprintf('Not a face\n');
% end; 
% 
% 
% min = 11e+5;
% min_index = 0;
% if (epsilon < 2.5e+05)
%     for i=1:m,
%         eps = sum((Y_in_eigenspace - Z(i,:)).^2);
%         if (eps < min)
%             min = eps;
%             min_index = i;
%         end;
%     end;
%     if (min_index == 0)
%         fprintf('Image did not match with any image in the database\n\n');
%     else
%         fprintf('Image matched with index- \n\n');
%         min_index
%     end;
%     
% end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% Error percentage on the training set in detecting they are face or not
count =0;
for i=1:m,
    ep = sum((X_norm(i,:) - X_rec(i,:)).^2);
    if (ep < 2.5e+05)
        count = count +1;
    end;
end;
fprintf('Percantage accuracy in training set\n');
percentage_accuracy = (count/m)*100;   %(output was 97.5145)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


