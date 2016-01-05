/********************************************************************/
/**Author: Xiaoshan Yang
/**Email: 000yangxiaoshan@163.com
/**University: Beijing Institute of Technology
/********************************************************************/

//论文算法实现
//Poisson image editing
//Patrick Pérez, M. Gangnet, A. Blake
//ACM Transactions on Graphics (SIGGRAPH'03), 22(3):313-318, 2003

#include "stdafx.h"
#include "string.h"
#include "cv.h"
#include "highgui.h"
#include "Poisson.h"

int main(int argc, char * argv[])
{
	if(argc == 1){
		printf("\nPoisson Image Editing:\n\n");
		printf("Usage:\n\n");
		printf("PoissonImageEditing [-decolor] [s_img] [s_mask] [rho] [iterN] [o_img]\n");
		printf("--Decolorization of the background\n");
		printf("----s_img: source image\n");
		printf("----s_mask: mask of the foreground pixels\n");
		printf("----rho: relax factor of the Gauss-seidel solver\n");
		printf("----iterN: number of the iteration for the Gauss-seidel solver\n");
		printf("----o_img: output image\n\n");
		printf("PoissonImageEditing [-til] [s_img] [Nx] [Ny] [rho] [iterN] [o_img]\n");
		printf("--Seamless tiling from a source image\n");
		printf("----s_img: source image\n");
		printf("----Nx: number of  replication in x direction\n");
		printf("----Ny: number of  replication in y direction\n");
		printf("----rho: relax factor of the Gauss-seidel solver\n");
		printf("----iterN: number of the iteration for the Gauss-seidel solver\n");
		printf("----o_img: output image\n\n");
		printf("PoissonImageEditing [-illum] [s_img] [s_mask] [alpha] [beta] [rho] [iterN] [o_img]\n");
		printf("--Local illumination changes of the selected region\n");
		printf("----s_img: source image\n");
		printf("----s_mask: mask of the selected region in the source image\n");
		printf("----alpha: parameter of  Eq(16) in the paper\n");
		printf("----beta: parameter of  Eq(16) in the paper\n");
		printf("----rho: relax factor of the Gauss-seidel solver\n");
		printf("----iterN: number of the iteration for the Gauss-seidel solver\n");
		printf("----o_img: output image\n\n");
		printf("PoissonImageEditing [-flat] [s_img] [s_mask] [threshold1] [threshold2] [rho] [iterN] [o_img]\n");
		printf("--Texture flattening of the selected region\n");
		printf("----s_img: source image\n");
		printf("----s_mask: mask of the selected region in the source image\n");
		printf("----threshold1: first threshold for the Canny edge detection algorithm\n");
		printf("----threshold2: second threshold for the Canny edge detection algorithm\n");
		printf("----rho: relax factor of the Gauss-seidel solver\n");
		printf("----iterN: number of the iteration for the Gauss-seidel solver\n");
		printf("----o_img: output image\n\n");
		printf("PoissonImageEditing [-import] [s_img] [s_mask] [t_img] [x] [y] [rho] [iterN] [o_img]\n");
		printf("--Seamless cloning with importing gradients\n");
		printf("----s_img: source image\n");
		printf("----s_mask: mask of the selected region in the source image\n");
		printf("----t_img: target image\n");
		printf("----x: x offset of the source pixels in target image\n");
		printf("----y: y offset of the source pixels in target image\n");
		printf("----rho: relax factor of the Gauss-seidel solver\n");
		printf("----iterN: number of the iteration for the Gauss-seidel solver\n");
		printf("----o_img: output image\n\n");
		printf("PoissonImageEditing [-mix] [s_img] [s_mask] [t_img] [x] [y] [rho] [iterN] [o_img]\n");
		printf("--Seamless cloning with mixing gradients\n");
		printf("----s_img: source image\n");
		printf("----s_mask: mask of selected region in the source image\n");
		printf("----t_img: target image\n");
		printf("----x: x offset of the source pixels in target image\n");
		printf("----y: y offset of the source pixels in target image\n");
		printf("----rho: relax factor of the Gauss-seidel solver\n");
		printf("----iterN: number of the iteration for the Gauss-seidel solver\n");
		printf("----o_img: output image\n\n");
		printf("PoissonImageEditing [-color] [s_img] [s_mask] [r_ratio] [g_ratio] [b_ratio] [rho] [iterN] [o_img]\n");
		printf("--Recoloring of the foreground\n");
		printf("----s_img: source image\n");
		printf("----s_mask: mask of the selected region in the source image\n");
		printf("----r_ratio: multiplying ratio for the R channel of the source pixels\n");
		printf("----g_ratio: multiplying ratio for the G channel of the source pixels\n");
		printf("----b_ratio: multiplying ratio for the B channel of the source pixels\n");
		printf("----rho: relax factor of the Gauss-seidel solver\n");
		printf("----iterN: number of the iteration for the Gauss-seidel solver\n");
		printf("----o_img: output image\n\n");
		system("pause");
	}
	else if(argc == 7){
		if(!strcmp(argv[1],"-decolor")){
			Poisson p(argv[2],argv[3]);
			cv::Mat o_img = p.PoissonDeColor(atof(argv[4]),atoi(argv[5]));
			cv::imwrite(argv[6],o_img);
			system("pause");
		}
	}
	else if(argc == 8){
		if(!strcmp(argv[1],"-til")){
			Poisson p(argv[2]);
			cv::Mat o_img = p.PoissonTil(atoi(argv[3]),atoi(argv[4]),atof(argv[5]),atoi(argv[6]));
			cv::imwrite(argv[7],o_img);
			system("pause");
		}
	}
	else if(argc == 9){
		if(!strcmp(argv[1],"-illum")){
			Poisson p(argv[2],argv[3]);
			cv::Mat o_img = p.PoissonIllum(atof(argv[4]),atof(argv[5]),0.000001f,atof(argv[6]),atoi(argv[7]));
			cv::imwrite(argv[8],o_img);
			system("pause");
		}
		if(!strcmp(argv[1],"-flat")){
			Poisson p(argv[2],argv[3]);
			cv::Mat o_img = p.PoissonFlat(atof(argv[4]),atof(argv[5]),atof(argv[6]),atoi(argv[7]));
			cv::imwrite(argv[8],o_img);
			system("pause");
		}
	}
	else if(argc == 10){
		if(!strcmp(argv[1],"-import")){
			Poisson p(argv[2],argv[3],argv[4]);
			cv::Mat o_img = p.PoissonImportCloning(atoi(argv[5]),atoi(argv[6]),atof(argv[7]),atoi(argv[8]));
			cv::imwrite(argv[9],o_img);
			system("pause");
		}
		else if(!strcmp(argv[1],"-mix")){
			Poisson p(argv[2],argv[3],argv[4]);
			cv::Mat o_img = p.PoissonMixCloning(atoi(argv[5]),atoi(argv[6]),atof(argv[7]),atoi(argv[8]));
			cv::imwrite(argv[9],o_img);
			system("pause");
		}
		if(!strcmp(argv[1],"-color")){
			Poisson p(argv[2],argv[3]);
			cv::Mat o_img = p.PoissonColor(atof(argv[4]),atof(argv[5]),atof(argv[6]),atof(argv[7]),atoi(argv[8]));
			cv::imwrite(argv[9],o_img);
			system("pause");
		}
	}
	else{
		exit(0);
	}	
	return 0;
}