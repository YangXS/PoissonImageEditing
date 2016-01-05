// PoissonImageEditing.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "string.h"
#include "cv.h"
#include "highgui.h"

const float rho = 1.9;      //高斯赛德尔迭代法的松弛因子
const int iterN = 2000;      //高斯赛德尔迭代法的迭代次数
const float alpha = 0.2;    //论文中(16)式对应的参数
const float beta = 0.2;     //论文中(16)式对应的参数
const float eps = 0.000001; //防止论文(16)式对应的有负指数的项的基数部分是0
const float s = 0.5;        //由亮度变换计算颜色变化的参数

//定义存取cv::Mat类型matrix的函数(使用这些函数简便，程序可读性好，不使用此函数，直接存取速度快：
inline uchar get8(cv::Mat src,int i,int j,int channel)
{
	uchar * d = (uchar*)(src.data + i*src.step);
	return d[j*src.channels()+channel];
}
inline void set8(cv::Mat src,int i,int j,int channel,uchar value)
{
	uchar * d = (uchar*)(src.data + i*src.step);
	d[j*src.channels()+channel] = value;
}
inline float get32(cv::Mat src,int i,int j,int channel)
{
	float * d = (float*)(src.data + i*src.step);
	return d[j*src.channels()+channel];
}
inline void set32(cv::Mat src,int i,int j,int channel,float value)
{
	float * d = (float*)(src.data + i*src.step);
	d[j*src.channels()+channel] = value;
}


cv::Mat PoissonImportCloning(cv::Mat s_img, cv::Mat s_mask, cv::Mat t_img, int x, int y);
cv::Mat PoissonMixCloning(cv::Mat s_img, cv::Mat s_mask, cv::Mat t_img, int x, int y);
cv::Mat PoissonFlat(cv::Mat s_img,cv::Mat gray_img,cv::Mat s_mask);
cv::Mat PoissonIllum(cv::Mat s_img,cv::Mat s_mask,double ratio);
cv::Mat PoissonColor(cv::Mat s_img,cv::Mat s_mask,double r_ratio,double g_ratio,double b_ratio);
cv::Mat PoissonDeColor(cv::Mat s_img,cv::Mat s_mask);
cv::Mat PoissonTil(cv::Mat s_img,int Nx,int Ny);

int main(int argc, char * argv[])
{
	if(argc == 1){
		printf("Implement of Poisson Image Editing:\n");
		printf("Usage:\n");
		printf("PoissonImageEditing [-import] [s_img] [s_mask] [t_img] [x] [y] [o_img]\n");
		printf("PoissonImageEditing [-mix] [s_img] [s_mask] [t_img] [x] [y] [o_img]\n");
		printf("PoissonImageEditing [-flat] [s_img] [s_mask] [o_img]\n");
		printf("Parameter explain:\n");
		printf("-cloning: denotes the cloning a set of pixels from source image to target image\n");
		printf("s_img: source image\n");
		printf("s_mask: binary mask for source image\n");
		printf("t_img: target image\n");
		printf("x: x offset of pixels in target image\n");
		printf("y: y offset of pixels in target image\n");
		printf("o_img: output image\n");
		//system("pause");
	}
	else if(argc == 6){
		if(!strcmp(argv[1],"-til")){
			cv::Mat s_img = cv::imread(argv[2]);
			//cv::Mat s_mask = cv::imread(argv[3],0);
			cv::Mat o_img = PoissonTil(s_img,atoi(argv[3]),atoi(argv[4]));
			cv::imwrite(argv[5],o_img);
		}
	}
	else if(argc == 8){
		if(!strcmp(argv[1],"-import")){
			//printf(argv[1]);
			system("pause");
			cv::Mat s_img = cv::imread(argv[2]);
			cv::Mat s_mask = cv::imread(argv[3],0);
			cv::Mat t_img = cv::imread(argv[4]);
			int x = atoi(argv[5]);
			int y = atoi(argv[6]);
			cv::Mat o_img = PoissonImportCloning(s_img,s_mask,t_img,x,y);
			cv::imwrite(argv[7],o_img);
		}
		else if(!strcmp(argv[1],"-mix")){
			system("pause");
			cv::Mat s_img = cv::imread(argv[2]);
			cv::Mat s_mask = cv::imread(argv[3],0);
			cv::Mat t_img = cv::imread(argv[4]);
			int x = atoi(argv[5]);
			int y = atoi(argv[6]);
			cv::Mat o_img = PoissonMixCloning(s_img,s_mask,t_img,x,y);
			cv::imwrite(argv[7],o_img);
		}
		if(!strcmp(argv[1],"-color")){
			cv::Mat s_img = cv::imread(argv[2]);
			//cv::Mat gray_img = cv::imread(argv[2],0);
			cv::Mat s_mask = cv::imread(argv[3],0);
			cv::Mat o_img = PoissonColor(s_img,s_mask,atof(argv[4]),atof(argv[5]),atof(argv[6]));
			cv::imwrite(argv[7],o_img);
			//system("pause");
		}
		//printf(argv[1]);
		//system("pause");
	}
	else if(argc == 5){
		if(!strcmp(argv[1],"-flat")){
			cv::Mat s_img = cv::imread(argv[2]);
			cv::Mat gray_img = cv::imread(argv[2],0);  //用于边缘检测
			cv::Mat s_mask = cv::imread(argv[3],0);
			cv::Mat o_img = PoissonFlat(s_img,gray_img,s_mask);
			cv::imwrite(argv[4],o_img);
		}
		if(!strcmp(argv[1],"-decolor")){
			cv::Mat s_img = cv::imread(argv[2]);
			cv::Mat s_mask = cv::imread(argv[3],0);
			cv::Mat o_img = PoissonDeColor(s_img,s_mask);
			cv::imwrite(argv[4],o_img);
		}
	}
	else if(argc == 6){
		if(!strcmp(argv[1],"-illum")){
			cv::Mat s_img = cv::imread(argv[2]);
			//cv::Mat gray_img = cv::imread(argv[2],0);
			cv::Mat s_mask = cv::imread(argv[3],0);
			cv::Mat o_img = PoissonIllum(s_img,s_mask,atof(argv[4]));
			cv::imwrite(argv[5],o_img);
			//system("pause");
		}
	}
	else{
		//system("pause");
		exit(0);
	}
	
	return 0;
}

cv::Mat PoissonImportCloning(cv::Mat s_img, cv::Mat s_mask, cv::Mat t_img, int x, int y)
{
	/*if(s_mask.type() != CV_8UC1){
		printf("error: mask must be 1 channel and 8 byte!\n");
		system("pause");
		exit(0);
	}*/
	//cv::Mat sm(s_mask.rows,s_mask.cols,CV_8UC1);
	s_mask.convertTo(s_mask,CV_8UC1);  //注意：如果imread()第二个参数不设为0，这里转换后的s_mask仍然是3通道；
	//printf("c:%d\n",s_mask.channels());
	//s_mask = s_m;
	//cv::namedWindow("show");
	//cv::imshow("show",s_mask);
	//cv::waitKey();
	//for(int i =0;i < s
	//cv::imwrite("mask.bmp",s_mask);
	s_img.convertTo(s_img,CV_32FC3);
	t_img.convertTo(t_img,CV_32FC3);
	cv::Mat o_img;
	t_img.copyTo(o_img);
	/*for(int i = 0;i < s_mask.rows;i++){
		for(int j = 0;j < s_mask.cols;j++){
			if(get8(s_mask,i,j,0)){
				for(int channel = 0;channel < o_img.channels();channel++){
					set32(o_img,i+y,j+x,channel,150);
				}
			}
		}
	}*/
	printf("start Gaussian Siedal iteration:\n");
	for(int iter = 0;iter < iterN;iter++){
		//printf("iter:%d\n",iter);
		for(int channel = 0;channel < o_img.channels();channel++){
		
			for(int i = 0;i < s_mask.rows;i++){
				//uchar * dM1 = (uchar*)(s_mask.data + (i-1)*s_mask.step);
				//uchar * dM2 = (uchar*)(s_mask.data + i*s_mask.step);
				//uchar * dM3 = (uchar*)(s_mask.data + (i+1)*s_mask.step);
				//float * dS1 = (float*)(s_img.data + (i-1)*s_img.step);
				//float * dS2 = (float*)(s_img.data + i*s_img.step);
				for(int j = 0;j < s_mask.cols;j++){
					if(get8(s_mask,i,j,0) == 255){
						//根据论文中的(7)式来迭代更新mask指定的像素值
						//dM2[j*sm.channels()] = 100;
						//dM2[j] == 255
						int Np = 0;
						float sum = 0;
						//求四邻域的像素和
						if((j-1) >= 0 && (j-1)+x >= 0){ //判断边界
							++Np;
							if(get8(s_mask,i,j-1,0)){   //位于omega内的邻域点
								sum += get32(o_img,i+y,j-1+x,channel);
							}
							else{ //位于边界上的邻域点
								sum += get32(t_img,i+y,j-1+x,channel);
							}
							sum += (get32(s_img,i,j,channel) - get32(s_img,i,j-1,channel));
						}
						if((j+1) < s_img.cols && (j+1)+x < o_img.cols){ 
							++Np;
							if(get8(s_mask,i,j+1,0)){
								sum += get32(o_img,i+y,j+1+x,channel);
							}
							else{
								sum += get32(t_img,i+y,j+1+x,channel);
							}
							sum += (get32(s_img,i,j,channel) - get32(s_img,i,j+1,channel));
						}
						if((i-1) >= 0 && (i-1)+y >= 0){
							++Np;
							if(get8(s_mask,i-1,j,0)){
								sum += get32(o_img,i-1+y,j+x,channel);
							}
							else{
								sum += get32(t_img,i-1+y,j+x,channel);
							}
							sum += (get32(s_img,i,j,channel) - get32(s_img,i-1,j,channel));
						}
						if((i+1) < s_img.rows && (i+1)+y < o_img.rows){
							++Np;
							if(get8(s_mask,i+1,j,0)){
								sum += get32(o_img,i+1+y,j+x,channel);
							}
							else{
								sum += get32(t_img,i+1+y,j+x,channel);
							}
							sum += (get32(s_img,i,j,channel) - get32(s_img,i+1,j,channel));
						}
						//for(int channel = 0;channel < o_img.channels();channel++){
						//dO = (float*)(o_img.data + (i+y)*o_img.step);
						//dO[(j+x)*o_img.channels()+channel] = (1-rho)*dO[(j+x)*o_img.channels()+channel] + rho/Np * sum;
						float newValue = (1-rho)*get32(o_img,i+y,j+x,channel) + rho/Np * sum;
						set32(o_img,i+y,j+x,channel,newValue);
						//dO[j*o_img.channels()+channel] = 0;
						//}
						//printf("i:%d，j:%d\nc:%d\n",i,j,sm.channels());
						//system("pause");
					}
				}
				
			}
		}
	}

	//cv::namedWindow("show");
	//cv::imshow("show",sm);
	//cv::waitKey();

	//o_img.convertTo(o_img,CV_8UC3);
	//cv::namedWindow("show");
	//cv::imshow("show",o_img);
	//cv::waitKey();
	return o_img;
}

cv::Mat PoissonMixCloning(cv::Mat s_img, cv::Mat s_mask, cv::Mat t_img, int x, int y)
{
	s_mask.convertTo(s_mask,CV_8UC1);  //注意：如果imread()第二个参数不设为0，这里转换后的s_mask仍然是3通道；
	//printf("c:%d\n",s_mask.channels());
	s_img.convertTo(s_img,CV_32FC3);
	t_img.convertTo(t_img,CV_32FC3);
	cv::Mat o_img;
	t_img.copyTo(o_img);
	printf("start Gaussian Siedal iteration:\n");
	for(int iter = 0;iter < iterN;iter++){
		//printf("iter:%d\n",iter);
		for(int channel = 0;channel < o_img.channels();channel++){		
			for(int i = 0;i < s_mask.rows;i++){
				for(int j = 0;j < s_mask.cols;j++){
					if(get8(s_mask,i,j,0)){
						int Np = 0;
						float sum = 0;
						float Vpq1 = 0;
						float Vpq2 = 0;
						//求四邻域的像素和
						if((j-1) >= 0 && (j-1)+x >= 0){ //判断边界
							++Np;
							if(get8(s_mask,i,j-1,0)){   //位于omega内的邻域点
								sum += get32(o_img,i+y,j-1+x,channel);
							}
							else{ //位于边界上的邻域点
								sum += get32(t_img,i+y,j-1+x,channel);
							}
							Vpq1 = (get32(t_img,i,j,channel) - get32(t_img,i,j-1,channel));
							Vpq2 = (get32(s_img,i,j,channel) - get32(s_img,i,j-1,channel));
							sum += ( abs(Vpq1) > abs(Vpq2) ? Vpq1 : Vpq2);
						}
						if((j+1) < s_img.cols && (j+1)+x < o_img.cols){ 
							++Np;
							if(get8(s_mask,i,j+1,0)){
								sum += get32(o_img,i+y,j+1+x,channel);
							}
							else{
								sum += get32(t_img,i+y,j+1+x,channel);
							}
							Vpq1 = (get32(t_img,i,j,channel) - get32(t_img,i,j+1,channel));
							Vpq2 = (get32(s_img,i,j,channel) - get32(s_img,i,j+1,channel));
							sum += ( abs(Vpq1) > abs(Vpq2) ? Vpq1 : Vpq2);
						}
						if((i-1) >= 0 && (i-1)+y >= 0){
							++Np;
							if(get8(s_mask,i-1,j,0)){
								sum += get32(o_img,i-1+y,j+x,channel);
							}
							else{
								sum += get32(t_img,i-1+y,j+x,channel);
							}
							Vpq1 = (get32(t_img,i,j,channel) - get32(t_img,i-1,j,channel));
							Vpq2 = (get32(s_img,i,j,channel) - get32(s_img,i-1,j,channel));
							sum += ( abs(Vpq1) > abs(Vpq2) ? Vpq1 : Vpq2);
						}
						if((i+1) < s_img.rows && (i+1)+y < o_img.rows){
							++Np;
							if(get8(s_mask,i+1,j,0)){
								sum += get32(o_img,i+1+y,j+x,channel);
							}
							else{
								sum += get32(t_img,i+1+y,j+x,channel);
							}
							Vpq1 = (get32(t_img,i,j,channel) - get32(t_img,i+1,j,channel));
							Vpq2 = (get32(s_img,i,j,channel) - get32(s_img,i+1,j,channel));
							sum += ( abs(Vpq1) > abs(Vpq2) ? Vpq1 : Vpq2);
						}
						//system("pause");
						float newValue = (1-rho)*get32(o_img,i+y,j+x,channel) + rho/Np * sum;
						set32(o_img,i+y,j+x,channel,newValue);
					}
				}
				//system("pause");
				
			}
			//system("pause");
		}
	}
	return o_img;
}

cv::Mat PoissonFlat(cv::Mat s_img,cv::Mat gray_img,cv::Mat s_mask)
{
	s_mask.convertTo(s_mask,CV_8UC1);  //注意：如果imread()第二个参数不设为0，这里转换后的s_mask仍然是3通道；
	//printf("c:%d\n",s_mask.channels());
	s_img.convertTo(s_img,CV_32FC3);
	cv::Mat o_img;
	s_img.copyTo(o_img);
	//cv::Mat gray_img(o_img.rows,o_img.cols,CV_8UC1);
	cv::Mat edge_img(s_img.rows,s_img.cols,CV_8UC1);
	cv::Canny(gray_img,edge_img,40,50);
	//cv::namedWindow("show");
	//cv::imshow("show",edge_img);
	//cv::waitKey();
	printf("start Gaussian Siedal iteration:\n");
	system("pause");
	for(int iter = 0;iter < iterN;iter++){
		//printf("iter:%d\n",iter);
		for(int channel = 0;channel < o_img.channels();channel++){		
			for(int i = 0;i < s_mask.rows;i++){
				for(int j = 0;j < s_mask.cols;j++){
					if(get8(s_mask,i,j,0)){
						int Np = 0;
						float sum = 0;
						//求四邻域的像素和
						if((j-1) >= 0){ //判断边界
							++Np;
							//if(get8(s_mask,i,j-1,0)){   //位于omega内的邻域点
							sum += get32(o_img,i,j-1,channel);
							//}
							//else{ //位于边界上的邻域点
								//sum += get32(t_img,i,j-1,channel);
							//}
							if(get8(edge_img,i,j,0) || get8(edge_img,i,j-1,0)){
								sum += (get32(s_img,i,j,channel) - get32(s_img,i,j-1,channel));
							}
						}
						if((j+1) < s_img.cols){ 
							++Np;
							//if(get8(s_mask,i,j+1,0)){
							sum += get32(o_img,i,j+1,channel);
							//}
							//else{
								//sum += get32(t_img,i+y,j+1+x,channel);
							//}
							if(get8(edge_img,i,j,0) || get8(edge_img,i,j+1,0)){
								sum += (get32(s_img,i,j,channel) - get32(s_img,i,j+1,channel));
							}
						}
						if((i-1) >= 0){
							++Np;
							//if(get8(s_mask,i-1,j,0)){
							sum += get32(o_img,i-1,j,channel);
							//}
							//else{
								//sum += get32(t_img,i-1+y,j+x,channel);
							//}
							if(get8(edge_img,i,j,0) || get8(edge_img,i-1,j,0)){
								sum += (get32(s_img,i,j,channel) - get32(s_img,i-1,j,channel));
							}
						}
						if((i+1) < s_img.rows){
							++Np;
							//if(get8(s_mask,i+1,j,0)){
							sum += get32(o_img,i+1,j,channel);
							//}
							//else{
								//sum += get32(t_img,i+1+y,j+x,channel);
							//}
							if(get8(edge_img,i,j,0) || get8(edge_img,i+1,j,0)){
								sum += (get32(s_img,i,j,channel) - get32(s_img,i+1,j,channel));
							}
						}
						//system("pause");
						float newValue = (1-rho)*get32(o_img,i,j,channel) + rho/Np * sum;
						set32(o_img,i,j,channel,newValue);
					}
				}
				//system("pause");
				
			}
			//system("pause");
		}
	}
	return o_img;
}

/*cv::Mat PoissonIllum(cv::Mat s_img,cv::Mat s_mask,double ratio)
{
	s_mask.convertTo(s_mask,CV_8UC1);  //注意：如果imread()第二个参数不设为0，这里转换后的s_mask仍然是3通道；
	//printf("c:%d\n",s_mask.channels());
	s_img.convertTo(s_img,CV_32FC3);
	//for(int i = 0;i < s_img.rows;i++){
		//for(int j = 0;j < s_img.cols;j++){
			//for(int channel = 0;channel < s_img.channels();channel++){
				//set32(s_img,i,j,channel,get32(s_img,i,j,channel));
			//}
		//}
	//}
	//cv::Mat o_img;
	//s_img.copyTo(o_img);

	cv::Mat log_Simg(s_img.rows,s_img.cols,s_img.type());
	cv::Mat log_Timg(s_img.rows,s_img.cols,s_img.type());
	cv::Mat log_Oimg(s_img.rows,s_img.cols,s_img.type());
	
	for(int i = 0;i < s_img.rows;i++){
		for(int j = 0;j < s_img.cols;j++){
			for(int channel = 0;channel < s_img.channels();channel++){
				float value = get32(s_img,i,j,channel);
				set32(log_Simg,i,j,channel,std::log(ratio*value+eps));
				set32(log_Timg,i,j,channel,std::log(value+eps));
				set32(log_Oimg,i,j,channel,std::log(value+eps));
			}
		}
	}

	//cv::namedWindow("show");
	//cv::imshow("show",edge_img);
	//cv::waitKey();
	printf("start Gaussian Siedal iteration:\n");
	system("pause");
	for(int iter = 0;iter < iterN;iter++){
		//printf("iter:%d\n",iter);
		for(int channel = 0;channel < log_Oimg.channels();channel++){		
			for(int i = 0;i < s_mask.rows;i++){
				for(int j = 0;j < s_mask.cols;j++){
					if(get8(s_mask,i,j,0)){
						int Np = 0;
						float sum = 0;
						float Fp = get32(log_Simg,i,j,channel);;
						float Fq;
						//求四邻域的像素和
						if((j-1) >= 0){ //判断边界
							++Np;							
							if(get8(s_mask,i,j-1,0)){   //位于omega内的邻域点	
								sum += get32(log_Oimg,i,j-1,channel);
							}
							else{ //位于边界上的邻域点
								sum += get32(log_Timg,i,j-1,channel);
							}
							Fq = get32(log_Simg,i,j-1,channel);
							sum += std::pow(alpha,beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
							//printf("Fp:%f\n",Fp);
							//printf("Fq:%f\n",Fq);
							//system("pause");
						}
						if((j+1) < s_img.cols){ 
							++Np;							
							if(get8(s_mask,i,j+1,0)){
								sum += get32(log_Oimg,i,j+1,channel);								
							}
							else{
								sum += get32(log_Timg,i,j+1,channel);
							}
							Fq = get32(log_Simg,i,j+1,channel);
							sum += std::pow(alpha,beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
							//printf("Fp:%f\n",Fp);
							//printf("Fq:%f\n",Fq);
							//system("pause");
						}
						if((i-1) >= 0){
							++Np;							
							if(get8(s_mask,i-1,j,0)){	
								sum += get32(log_Oimg,i-1,j,channel);								
							}
							else{
								sum += get32(log_Timg,i-1,j,channel);
							}
							Fq = get32(log_Simg,i-1,j,channel);
							sum += std::pow(alpha,beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
							//printf("Fp:%f\n",Fp);
							//printf("Fq:%f\n",Fq);
							//system("pause");
						}
						if((i+1) < s_img.rows){
							++Np;							
							if(get8(s_mask,i+1,j,0)){
								sum += get32(log_Oimg,i+1,j,channel);								
							}
							else{
								sum += get32(log_Timg,i+1,j,channel);
							}
							Fq = get32(log_Simg,i+1,j,channel);
							sum += std::pow(alpha,beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
							//if(get8(edge_img,i,j,0) || get8(edge_img,i+1,j,0)){
							//printf("Fp:%f\n",Fp);
							//printf("Fq:%f\n",Fq);
							//system("pause");
							//}
						}	
						//printf("Np:%d\n",Np);
						//printf("sum:%f\n",sum);
						//printf("value:%f\n",get32(o_img,i,j,channel));
						//system("pause");
						float newValue = (1-rho)*get32(log_Oimg,i,j,channel) + rho/Np * sum;
						set32(log_Oimg,i,j,channel,newValue);
					}
				}
				//system("pause");
				
			}
			//system("pause");
		}
	}
	for(int i = 0;i < log_Oimg.rows;i++){
		for(int j = 0;j < log_Oimg.cols;j++){
			for(int channel = 0;channel < log_Oimg.channels();channel++){
				set32(log_Oimg,i,j,channel,std::exp(get32(log_Oimg,i,j,channel)));
			}
		}
	}
	return log_Oimg;
}*/

cv::Mat PoissonIllum(cv::Mat s_img,cv::Mat s_mask,double ratio)
{
	s_mask.convertTo(s_mask,CV_8UC1);  //注意：如果imread()第二个参数不设为0，这里转换后的s_mask仍然是3通道；
	//printf("c:%d\n",s_mask.channels());
	s_img.convertTo(s_img,CV_32FC3);

	cv::Mat YUV_img(s_img.rows,s_img.cols,s_img.type());
	cv::Mat log_Simg(s_img.rows,s_img.cols,s_img.type());
	cv::Mat log_Timg(s_img.rows,s_img.cols,s_img.type());
	cv::Mat log_Oimg(s_img.rows,s_img.cols,s_img.type());
	cv::cvtColor(s_img,YUV_img,CV_BGR2YCrCb);

	for(int i = 0;i < s_img.rows;i++){
		for(int j = 0;j < s_img.cols;j++){
			for(int channel = 0;channel < 1;channel++){
				float value = get32(YUV_img,i,j,channel);
				set32(log_Timg,i,j,channel,std::log(value+eps));
				set32(log_Oimg,i,j,channel,std::log(value+eps));
				value = value*ratio;
				if(value > 255){
					value = 255;
				}
				set32(log_Simg,i,j,channel,std::log(value+eps));
			}
		}
	}

	//log_img.copyTo(o_img);
	//cv::namedWindow("show");
	//cv::imshow("show",log_img);
	//cv::waitKey();
	printf("start Gaussian Siedal iteration:\n");
	system("pause");
	for(int iter = 0;iter < iterN;iter++){
		//printf("iter:%d\n",iter);
		for(int channel = 0;channel < 1;channel++){		
			for(int i = 0;i < s_mask.rows;i++){
				for(int j = 0;j < s_mask.cols;j++){
					if(get8(s_mask,i,j,0)){
						int Np = 0;
						float sum = 0;
						float Fp = get32(log_Simg,i,j,channel);
						float Fq;
						//求四邻域的像素和
						if((j-1) >= 0){ //判断边界
							++Np;
							if(get8(s_mask,i,j-1,0)){   //位于omega内的邻域点
								sum += get32(log_Oimg,i,j-1,channel);							
								//sum += (Fp-Fq);
								//Fq = get32(log_img,i,j-1,channel);
								//sum += std::pow(alpha,beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
							}
							else{ //位于边界上的邻域点
								sum += get32(log_Timg,i,j-1,channel);
								//sum += 255;
								//Fq = get32(log_img,i,j-1,channel);
								//sum += std::pow(alpha,beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
							}
							Fq = get32(log_Simg,i,j-1,channel);
							sum += std::pow(alpha,beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
							//sum += (Fp-Fq);
							//printf("Fp:%f\n",Fp);
							//printf("Fq:%f\n",Fq);
							//system("pause");
						}
						if((j+1) < s_img.cols){ 
							++Np;							
							if(get8(s_mask,i,j+1,0)){		
								sum += get32(log_Oimg,i,j+1,channel);								
								//sum += (Fp-Fq);
								//Fq = get32(log_img,i,j+1,channel);
								//sum += std::pow(alpha,beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
							}
							else{
								sum += get32(log_Timg,i,j+1,channel);
								//sum += 255;
								//Fq = get32(log_img,i,j+1,channel);
								//sum += std::pow(alpha,beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
							}
							Fq = get32(log_Simg,i,j+1,channel);
							sum += std::pow(alpha,beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
							//sum += (Fp-Fq);
							//printf("Fp:%f\n",Fp);
							//printf("Fq:%f\n",Fq);
							//system("pause");
						}
						if((i-1) >= 0){
							++Np;							
							if(get8(s_mask,i-1,j,0)){
								sum += get32(log_Oimg,i-1,j,channel);								
								//sum += (Fp-Fq);
								//Fq = get32(log_img,i-1,j,channel);
								//sum += std::pow(alpha,beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
							}
							else{
								sum += get32(log_Timg,i-1,j,channel);
								//sum += 255;
								//Fq = get32(log_img,i-1,j,channel);
								//sum += std::pow(alpha,beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
							}
							Fq = get32(log_Simg,i-1,j,channel);
							sum += std::pow(alpha,beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
							//sum += (Fp-Fq);
							//printf("Fp:%f\n",Fp);
							//printf("Fq:%f\n",Fq);
							//system("pause");
						}
						if((i+1) < s_img.rows){
							++Np;							
							if(get8(s_mask,i+1,j,0)){
								sum += get32(log_Oimg,i+1,j,channel);								
								//sum += (Fp-Fq);
								//Fq = get32(log_img,i+1,j,channel);
								//sum += std::pow(alpha,beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
							}
							else{
								//printf("value:%f\n",get32(log_img,i+1,j,channel));
								//system("pause");
								sum += get32(log_Timg,i+1,j,channel);
								//sum += 255;
								//Fq = get32(log_img,i+1,j,channel);
								//sum += std::pow(alpha,beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
							}
							Fq = get32(log_Simg,i+1,j,channel);
							sum += std::pow(alpha,beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
							//sum += (Fp-Fq);
							//if(get8(edge_img,i,j,0) || get8(edge_img,i+1,j,0)){
							//printf("Fp:%f\n",Fp);
							//printf("Fq:%f\n",Fq);
							//system("pause");
							//}
						}	
						//printf("Np:%d\n",Np);
						//printf("sum:%f\n",sum);
						//printf("value:%f\n",get32(o_img,i,j,channel));
						//system("pause");
						float newValue = (1-rho)*get32(log_Oimg,i,j,channel) + rho/Np * sum;
						set32(log_Oimg,i,j,channel,newValue);
					}
				}
				//system("pause");
				
			}
			//system("pause");
		}
	}
	for(int i = 0;i < log_Oimg.rows;i++){
		for(int j = 0;j < log_Oimg.cols;j++){
			for(int channel = 0;channel < 1;channel++){
				set32(log_Oimg,i,j,channel,std::exp(get32(log_Oimg,i,j,channel)));
			}
		}
	}

	cv::Mat res_img(s_img.rows,s_img.cols,s_img.type());
	for(int i = 0;i < res_img.rows;i++){
		for(int j = 0;j < res_img.cols;j++){
			for(int channel = 0;channel < res_img.channels();channel++){
				set32(res_img,i,j,channel,get32(log_Oimg,i,j,0)*std::pow((get32(s_img,i,j,channel)/get32(YUV_img,i,j,0)),s));
			}
		}
	}
	//cv::Mat res;
	//cv::cvtColor(res_img,res_img,CV_YCrCb2BGR);
	return res_img;
	//return o_img;
}

cv::Mat PoissonDeColor(cv::Mat s_img,cv::Mat s_mask)     // rho = 1.9; 迭代1000次以上
{
	//cv::Mat sm(s_mask.rows,s_mask.cols,CV_8UC1);
	s_mask.convertTo(s_mask,CV_8UC1);  //注意：如果imread()第二个参数不设为0，这里转换后的s_mask仍然是3通道；
	//printf("c:%d\n",s_mask.channels());
	//s_mask = s_m;
	//for(int i =0;i < s
	//cv::imwrite("mask.bmp",s_mask);
	s_img.convertTo(s_img,CV_32FC3);
	cv::Mat t_img(s_img.rows,s_img.cols,s_img.type());
	cv::Mat o_img(t_img.rows,t_img.cols,t_img.type());
	cv::Mat gray_img;
	cv::cvtColor(s_img,gray_img,CV_BGR2GRAY);
	for(int i = 0;i < t_img.rows;i++){
		for(int j = 0;j < t_img.cols;j++){
			float value = get32(gray_img,i,j,0);
			set32(t_img,i,j,0,value);
			set32(t_img,i,j,1,value);
			set32(t_img,i,j,2,value);
		}
	}
	t_img.copyTo(o_img);

	//printf("c:%d\n",o_img.channels());
	//system("pause");

	//s_img.convertTo(s_img,CV_8UC3);
	//cv::namedWindow("show");
	//cv::imshow("show",s_img);
	//cv::waitKey();
	//t_img.convertTo(t_img,CV_8UC3);
	//cv::imshow("show",t_img);
	//cv::waitKey();
	//o_img.convertTo(o_img,CV_8UC3);
	//cv::imshow("show",o_img);
	//cv::waitKey();
	/*for(int i = 0;i < s_mask.rows;i++){
		for(int j = 0;j < s_mask.cols;j++){
			if(get8(s_mask,i,j,0)){
				for(int channel = 0;channel < o_img.channels();channel++){
					set32(o_img,i,j,channel,150);
				}
			}
		}
	}*/

	/*for(int i = 0;i < s_img.rows;i++){
		for(int j = 0;j < s_img.cols;j++){
			set32(s_img,i,j,0,b_ratio*get32(s_img,i,j,0));
			set32(s_img,i,j,1,g_ratio*get32(s_img,i,j,1));
			set32(s_img,i,j,2,r_ratio*get32(s_img,i,j,2));
		}
	}*/

	printf("start Gaussian Siedal iteration:\n");

	for(int iter = 0;iter < iterN;iter++){
		//printf("iter:%d\n",iter);
		for(int channel = 0;channel < o_img.channels();channel++){
		
			for(int i = 0;i < s_mask.rows;i++){
				//uchar * dM1 = (uchar*)(s_mask.data + (i-1)*s_mask.step);
				//uchar * dM2 = (uchar*)(s_mask.data + i*s_mask.step);
				//uchar * dM3 = (uchar*)(s_mask.data + (i+1)*s_mask.step);
				//float * dS1 = (float*)(s_img.data + (i-1)*s_img.step);
				//float * dS2 = (float*)(s_img.data + i*s_img.step);
				for(int j = 0;j < s_mask.cols;j++){
					if(get8(s_mask,i,j,0) == 255){
						//根据论文中的(7)式来迭代更新mask指定的像素值
						//dM2[j*sm.channels()] = 100;
						//dM2[j] == 255
						int Np = 0;
						float sum = 0;
						//求四邻域的像素和
						if((j-1) >= 0){ //判断边界
							++Np;
							if(get8(s_mask,i,j-1,0)){   //位于omega内的邻域点
								sum += get32(o_img,i,j-1,channel);
							}
							else{ //位于边界上的邻域点
								sum += get32(t_img,i,j-1,channel);
							}
							sum += (get32(s_img,i,j,channel) - get32(s_img,i,j-1,channel));
						}
						if((j+1) < s_img.cols){ 
							++Np;
							if(get8(s_mask,i,j+1,0)){
								sum += get32(o_img,i,j+1,channel);
							}
							else{
								sum += get32(t_img,i,j+1,channel);
							}
							sum += (get32(s_img,i,j,channel) - get32(s_img,i,j+1,channel));
						}
						if((i-1) >= 0){
							++Np;
							if(get8(s_mask,i-1,j,0)){
								sum += get32(o_img,i-1,j,channel);
							}
							else{
								sum += get32(t_img,i-1,j,channel);
							}
							sum += (get32(s_img,i,j,channel) - get32(s_img,i-1,j,channel));
						}
						if((i+1) < s_img.rows){
							++Np;
							if(get8(s_mask,i+1,j,0)){
								sum += get32(o_img,i+1,j,channel);
							}
							else{
								sum += get32(t_img,i+1,j,channel);
							}
							sum += (get32(s_img,i,j,channel) - get32(s_img,i+1,j,channel));
						}
						//for(int channel = 0;channel < o_img.channels();channel++){
						//dO = (float*)(o_img.data + (i+y)*o_img.step);
						//dO[(j+x)*o_img.channels()+channel] = (1-rho)*dO[(j+x)*o_img.channels()+channel] + rho/Np * sum;
						float newValue = (1-rho)*get32(o_img,i,j,channel) + rho/Np * sum;
						set32(o_img,i,j,channel,newValue);
						//dO[j*o_img.channels()+channel] = 0;
						//}
						//printf("i:%d，j:%d\nc:%d\n",i,j,sm.channels());
						//system("pause");
					}
				}
				
			}
		}
	}

	//cv::namedWindow("show");
	//cv::imshow("show",sm);
	//cv::waitKey();

	//o_img.convertTo(o_img,CV_8UC3);
	//cv::namedWindow("show");
	//cv::imshow("show",o_img);
	//cv::waitKey();
	return o_img;

}

cv::Mat PoissonColor(cv::Mat s_img,cv::Mat s_mask,double r_ratio,double g_ratio,double b_ratio)   // rho = 1.9; 迭代1000次以上
{
	//cv::Mat sm(s_mask.rows,s_mask.cols,CV_8UC1);
	s_mask.convertTo(s_mask,CV_8UC1);  //注意：如果imread()第二个参数不设为0，这里转换后的s_mask仍然是3通道；
	//printf("c:%d\n",s_mask.channels());
	//s_mask = s_m;
	//cv::namedWindow("show");
	//cv::imshow("show",s_mask);
	//cv::waitKey();
	//for(int i =0;i < s
	//cv::imwrite("mask.bmp",s_mask);
	s_img.convertTo(s_img,CV_32FC3);
	//t_img.convertTo(t_img,CV_32FC3);
	cv::Mat t_img;
	s_img.copyTo(t_img);
	cv::Mat o_img(s_img.rows,s_img.cols,s_img.type());
	s_img.copyTo(o_img);
	/*for(int i = 0;i < s_mask.rows;i++){
		for(int j = 0;j < s_mask.cols;j++){
			if(get8(s_mask,i,j,0)){
				for(int channel = 0;channel < o_img.channels();channel++){
					set32(o_img,i,j,channel,150);
				}
			}
		}
	}*/

	for(int i = 0;i < s_img.rows;i++){
		for(int j = 0;j < s_img.cols;j++){
			set32(s_img,i,j,0,b_ratio*get32(s_img,i,j,0));
			set32(s_img,i,j,1,g_ratio*get32(s_img,i,j,1));
			set32(s_img,i,j,2,r_ratio*get32(s_img,i,j,2));
		}
	}

	printf("start Gaussian Siedal iteration:\n");

	for(int iter = 0;iter < iterN;iter++){
		//printf("iter:%d\n",iter);
		for(int channel = 0;channel < o_img.channels();channel++){
		
			for(int i = 0;i < s_mask.rows;i++){
				//uchar * dM1 = (uchar*)(s_mask.data + (i-1)*s_mask.step);
				//uchar * dM2 = (uchar*)(s_mask.data + i*s_mask.step);
				//uchar * dM3 = (uchar*)(s_mask.data + (i+1)*s_mask.step);
				//float * dS1 = (float*)(s_img.data + (i-1)*s_img.step);
				//float * dS2 = (float*)(s_img.data + i*s_img.step);
				for(int j = 0;j < s_mask.cols;j++){
					if(get8(s_mask,i,j,0) == 255){
						//根据论文中的(7)式来迭代更新mask指定的像素值
						//dM2[j*sm.channels()] = 100;
						//dM2[j] == 255
						int Np = 0;
						float sum = 0;
						//求四邻域的像素和
						if((j-1) >= 0){ //判断边界
							++Np;
							if(get8(s_mask,i,j-1,0)){   //位于omega内的邻域点
								sum += get32(o_img,i,j-1,channel);
							}
							else{ //位于边界上的邻域点
								sum += get32(t_img,i,j-1,channel);
							}
							sum += (get32(s_img,i,j,channel) - get32(s_img,i,j-1,channel));
						}
						if((j+1) < s_img.cols){ 
							++Np;
							if(get8(s_mask,i,j+1,0)){
								sum += get32(o_img,i,j+1,channel);
							}
							else{
								sum += get32(t_img,i,j+1,channel);
							}
							sum += (get32(s_img,i,j,channel) - get32(s_img,i,j+1,channel));
						}
						if((i-1) >= 0){
							++Np;
							if(get8(s_mask,i-1,j,0)){
								sum += get32(o_img,i-1,j,channel);
							}
							else{
								sum += get32(t_img,i-1,j,channel);
							}
							sum += (get32(s_img,i,j,channel) - get32(s_img,i-1,j,channel));
						}
						if((i+1) < s_img.rows){
							++Np;
							if(get8(s_mask,i+1,j,0)){
								sum += get32(o_img,i+1,j,channel);
							}
							else{
								sum += get32(t_img,i+1,j,channel);
							}
							sum += (get32(s_img,i,j,channel) - get32(s_img,i+1,j,channel));
						}
						//for(int channel = 0;channel < o_img.channels();channel++){
						//dO = (float*)(o_img.data + (i+y)*o_img.step);
						//dO[(j+x)*o_img.channels()+channel] = (1-rho)*dO[(j+x)*o_img.channels()+channel] + rho/Np * sum;
						float newValue = (1-rho)*get32(o_img,i,j,channel) + rho/Np * sum;
						set32(o_img,i,j,channel,newValue);
						//dO[j*o_img.channels()+channel] = 0;
						//}
						//printf("i:%d，j:%d\nc:%d\n",i,j,sm.channels());
						//system("pause");
					}
				}
				
			}
		}
	}

	//cv::namedWindow("show");
	//cv::imshow("show",sm);
	//cv::waitKey();

	//o_img.convertTo(o_img,CV_8UC3);
	//cv::namedWindow("show");
	//cv::imshow("show",o_img);
	//cv::waitKey();
	return o_img;

}

cv::Mat PoissonTil(cv::Mat s_img,int Nx,int Ny)     // rho = 1.9; 迭代1000次以上
{   
	//Nx和Ny分别是原始图像分别在列和行方向上的重复数目

	//cv::Mat sm(s_mask.rows,s_mask.cols,CV_8UC1);
	//s_mask.convertTo(s_mask,CV_8UC1);  //注意：如果imread()第二个参数不设为0，这里转换后的s_mask仍然是3通道；
	//printf("c:%d\n",s_mask.channels());
	//s_mask = s_m;
	//for(int i =0;i < s
	//cv::imwrite("mask.bmp",s_mask);
	s_img.convertTo(s_img,CV_32FC3);
	cv::Mat t_img(s_img.rows*Ny-(Ny-1),s_img.cols*Nx-(Nx-1),s_img.type());
	cv::Mat o_img(t_img.rows,t_img.cols,t_img.type());
	//cv::Mat gray_img;
	//cv::cvtColor(s_img,gray_img,CV_BGR2GRAY);
	for(int y = 0;y < Ny;y++){
		for(int x = 0;x < Nx;x++){
			for(int i = 0;i < s_img.rows;i++){
				for(int j = 0;j < s_img.cols;j++){
					for(int c = 0;c < s_img.channels();c++){
						float Value = get32(s_img,i,j,c);	
						if(i == 0 || i == s_img.rows-1){
							Value = 0.5*(get32(s_img,0,j,c)+get32(s_img,s_img.rows-1,j,c));
						}
						if(j == 0 || j == s_img.cols-1){
							Value = 0.5*(get32(s_img,i,0,c)+get32(s_img,i,s_img.cols-1,c));
						}
						set32(t_img,y*s_img.rows-y+i,x*s_img.cols-x+j,c,Value);
					}
				}
			}
		}
	}
	t_img.copyTo(o_img);
	//cv::namedWindow("show");
	//cv::imshow("show",o_img);
	//cv::waitKey();

	//printf("c:%d\n",o_img.channels());
	//system("pause");

	//s_img.convertTo(s_img,CV_8UC3);
	//cv::namedWindow("show");
	//cv::imshow("show",s_img);
	//cv::waitKey();
	//t_img.convertTo(t_img,CV_8UC3);
	//cv::imshow("show",t_img);
	//cv::waitKey();
	//o_img.convertTo(o_img,CV_8UC3);
	//cv::imshow("show",o_img);
	//cv::waitKey();
	/*for(int i = 0;i < s_mask.rows;i++){
		for(int j = 0;j < s_mask.cols;j++){
			if(get8(s_mask,i,j,0)){
				for(int channel = 0;channel < o_img.channels();channel++){
					set32(o_img,i,j,channel,150);
				}
			}
		}
	}*/

	/*for(int i = 0;i < s_img.rows;i++){
		for(int j = 0;j < s_img.cols;j++){
			set32(s_img,i,j,0,b_ratio*get32(s_img,i,j,0));
			set32(s_img,i,j,1,g_ratio*get32(s_img,i,j,1));
			set32(s_img,i,j,2,r_ratio*get32(s_img,i,j,2));
		}
	}*/

	printf("start Gaussian Siedal iteration:\n");

	for(int iter = 0;iter < iterN;iter++){
		//printf("iter:%d\n",iter);
		for(int y = 0;y < Ny;y++){
			for(int x = 0;x < Nx;x++){

				for(int channel = 0;channel < s_img.channels();channel++){
				
					for(int i = 0;i < s_img.rows;i++){
						//uchar * dM1 = (uchar*)(s_mask.data + (i-1)*s_mask.step);
						//uchar * dM2 = (uchar*)(s_mask.data + i*s_mask.step);
						//uchar * dM3 = (uchar*)(s_mask.data + (i+1)*s_mask.step);
						//float * dS1 = (float*)(s_img.data + (i-1)*s_img.step);
						//float * dS2 = (float*)(s_img.data + i*s_img.step);
						for(int j = 0;j < s_img.cols;j++){
							if(i > 0 && i < s_img.rows-1 && j > 0 && j < s_img.cols-1){
								//根据论文中的(7)式来迭代更新mask指定的像素值
								//dM2[j*sm.channels()] = 100;
								//dM2[j] == 255
								int Np = 0;
								float sum = 0;
								//求四邻域的像素和
								if((j-1) >= 0){ //判断边界
									++Np;
									if(i > 0 && i < s_img.rows-1 && j-1 > 0 && j-1 < s_img.cols-1){   //位于omega内的邻域点
										sum += get32(o_img,(y*s_img.rows-y)+i,(x*s_img.cols-x)+(j-1),channel);
									}
									else{ //位于边界上的邻域点
										sum += get32(t_img,(y*s_img.rows-y)+i,(x*s_img.cols-x)+(j-1),channel);
									}
									sum += (get32(s_img,i,j,channel) - get32(s_img,i,j-1,channel));
								}
								if((j+1) < s_img.cols){ 
									++Np;
									if(i > 0 && i < s_img.rows-1 && j+1 > 0 && j+1 < s_img.cols-1){
										sum += get32(o_img,(y*s_img.rows-y)+i,(x*s_img.cols-x)+(j+1),channel);
									}
									else{
										sum += get32(t_img,(y*s_img.rows-y)+i,(x*s_img.cols-x)+(j+1),channel);
									}
									sum += (get32(s_img,i,j,channel) - get32(s_img,i,j+1,channel));
								}
								if((i-1) >= 0){
									++Np;
									if(i-1 > 0 && i-1 < s_img.rows-1 && j > 0 && j < s_img.cols-1){
										sum += get32(o_img,(y*s_img.rows-y)+(i-1),(x*s_img.cols-x)+j,channel);
									}
									else{
										sum += get32(t_img,(y*s_img.rows-y)+(i-1),(x*s_img.cols-x)+j,channel);
									}
									sum += (get32(s_img,i,j,channel) - get32(s_img,i-1,j,channel));
								}
								if((i+1) < s_img.rows){
									++Np;
									if(i+1 > 0 && i+1 < s_img.rows-1 && j > 0 && j < s_img.cols-1){
										sum += get32(o_img,(y*s_img.rows-y)+(i+1),(x*s_img.cols-x)+j,channel);
									}
									else{
										sum += get32(t_img,(y*s_img.rows-y)+(i+1),(x*s_img.cols-x)+j,channel);
									}
									sum += (get32(s_img,i,j,channel) - get32(s_img,i+1,j,channel));
								}
								//for(int channel = 0;channel < o_img.channels();channel++){
								//dO = (float*)(o_img.data + (i+y)*o_img.step);
								//dO[(j+x)*o_img.channels()+channel] = (1-rho)*dO[(j+x)*o_img.channels()+channel] + rho/Np * sum;
								float newValue = (1-rho)*get32(o_img,(y*s_img.rows-y)+i,(x*s_img.cols-x)+j,channel) + rho/Np * sum;
								set32(o_img,(y*s_img.rows-y)+i,(x*s_img.cols-x)+j,channel,newValue);
								//dO[j*o_img.channels()+channel] = 0;
								//}
								//printf("i:%d，j:%d\nc:%d\n",i,j,sm.channels());
								//system("pause");
							}
						}
						
					}
				}//channel
			}
		}
	}

	//cv::namedWindow("show");
	//cv::imshow("show",sm);
	//cv::waitKey();

	//o_img.convertTo(o_img,CV_8UC3);
	//cv::namedWindow("show");
	//cv::imshow("show",o_img);
	//cv::waitKey();
	return o_img;

}