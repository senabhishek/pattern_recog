function a = my_rep(m)
image_size = [16 16];
preproc = im_box([],0,1)*im_gauss*im_resize([],image_size)*im_box([],1,0); %im_gauss for blurring
data_scaled = m*preproc;
a = prdataset(data_scaled);
end