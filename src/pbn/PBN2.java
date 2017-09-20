package pbn;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class PBN2 {

	public static void main (String[] args) {		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		/*Mat img = Mat.zeros(200, 200, CvType.CV_8UC3);
		Core.rectangle(img, new Point(0, 0), new Point(200, 50), new Scalar(0, 255, 0), -1);
		Core.rectangle(img, new Point(0, 50), new Point(200, 100), new Scalar(0, 0, 255), -1);
		Core.rectangle(img, new Point(0, 100), new Point(200, 150), new Scalar(255, 255, 0), -1);
		Core.rectangle(img, new Point(0, 150), new Point(200, 200), new Scalar(255, 0, 0), -1);*/
		Mat img = Highgui.imread("tukan.jpg");
		List<Mat> clusters = cluster(img, 8);
		
	
		for(int i =0; i< clusters.size();i++){
			Highgui.imwrite("img_"+i+".png", clusters.get(i));	
		}
		Highgui.imwrite("original.jpg", img);
		Mat t = drawEdges(clusters);
	}
	
	public static List<Mat> cluster(Mat cutout, int k) {
		//Mat samplesLAB = new Mat();
		//Imgproc.cvtColor(cutout, samplesLAB, Imgproc.COLOR_BGR2Lab);
		Mat blurred = new Mat();
		Imgproc.GaussianBlur(cutout, blurred, new Size(1,1), 0);
		Mat samplesLAB =blurred;
		Mat samples = samplesLAB.reshape(1, samplesLAB.cols() * samplesLAB.rows());
		
		System.out.println(samples.toString());
		Mat samples32f = new Mat();
		samples.convertTo(samples32f, CvType.CV_32F);
		System.out.println(samples32f.toString());
		
		Mat labels = new Mat();
		TermCriteria criteria = new TermCriteria(TermCriteria.COUNT, 100, 1);
		Mat centers = new Mat();
		Core.kmeans(samples32f, k, labels, criteria, 1, Core.KMEANS_PP_CENTERS, centers);	
		Highgui.imwrite("centers.jpg", centers);
		System.out.println(labels.toString());
		return showClusters(samplesLAB, labels, centers);
	}

	private static List<Mat> showClusters (Mat cutout, Mat labels, Mat centers) {
		
		List<Mat> rel = new ArrayList<Mat>();
		Mat re = new Mat(cutout.size(),CvType.CV_8UC3);
		Mat l2 = labels.reshape(1, cutout.rows());
		System.out.println("labels"+l2.toString()+"cols"+l2.size());
		for(int i = 0; i < cutout.rows();i++){
			for(int j = 0; j < cutout.cols();j++){
				int label = (int) l2.get(i, j)[0];
				double colr0 = centers.get(label, 0)[0];
				double colr1 = centers.get(label, 1)[0];
				double colr2 = centers.get(label, 2)[0];
				double[] v = new double[]{colr0,colr1,colr2};
				re.put(i, j, v);
				
			}
		}		
		rel.add(re);
		
		for(int i= 0; i< centers.rows();i++){
			Mat part = Mat.zeros(cutout.size(),CvType.CV_8UC1);
			
			for(int x = 0; x < cutout.rows();x++){
				for(int y = 0; y < cutout.cols();y++){
					int label = (int) l2.get(x, y)[0];
					if(label==i){
						part.put(x, y, 255.0);//array
					}
				}
			}
			rel.add(part);
			
		}
		rel.add(centers);
		
		return rel;		
	}
	
	private static Mat drawEdges(List<Mat> clusters){
		Mat re = Mat.zeros(clusters.get(0).size(),CvType.CV_8UC1);
		for(int i =1; i< clusters.size()-1;i++){
			Mat c = clusters.get(i);
			List<MatOfPoint> points = new ArrayList<MatOfPoint>();
	        Mat hierarchy = new Mat();
	        Imgproc.findContours(c, points, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
			//note! image border will be clipped
	        
	        List<MatOfPoint> filteredPoints = new ArrayList<MatOfPoint>();
	        
	        //System.out.println("pointslength:"+points.size()+" hierarchy:"+hierarchy.cols()+" " +hierarchy.get(0, 0)[3]);
	       for(int k =0; k<points.size();k++){
	            MatOfPoint contour = points.get(k);
	            if (Imgproc.contourArea(contour) > 32) {
	                filteredPoints.add(contour);
	                if(hierarchy.get(0, k)[3]<0)
	                Core.putText(re, ""+i, new Point (contour.get(0,0)[0],contour.get(0,0)[1]+14), Core.FONT_HERSHEY_PLAIN, 1, new Scalar(100,100,100));
	            }
	        }

	        Mat cont = Mat.zeros(clusters.get(0).size(),CvType.CV_8UC1);
	        Imgproc.drawContours(cont, filteredPoints, -1, new Scalar(255.0));
	        Imgproc.drawContours(re, filteredPoints, -1, new Scalar(55.0));
	        
	      
	        /*Iterator<MatOfPoint> each2 = filteredPoints.iterator();
	        while (each2.hasNext()) {
	            MatOfPoint contour2 = each2.next();
	            Core.putText(re, ""+i, new Point (contour2.get(0,0)[0],contour2.get(0,0)[1]), Core.FONT_HERSHEY_PLAIN, 1, new Scalar(75,75,75));
	            Core.putText(cont, ""+i, new Point (contour2.get(0,0)[0],contour2.get(0,0)[1]), Core.FONT_HERSHEY_PLAIN, 1, new Scalar(255));
	        }*/
	       
	        Highgui.imwrite("cont_"+i+".png", cont);	
		}
		
		 Mat allwhite = Mat.ones(clusters.get(0).size(),CvType.CV_8UC1);
		 allwhite.setTo(new Scalar(255.));
		 Mat reInv = new Mat();
		 Core.subtract(allwhite,re,reInv);
		 Mat bott = Mat.zeros(new Size(reInv.cols(),50),CvType.CV_8UC3);
		 
		 Mat centers = clusters.get(clusters.size()-1);
		 int swidth = (int) (bott.cols()/centers.rows());
		 int rectwidth = swidth - 20;
		 for (int l = 0; l< centers.rows();l++){
			 double col1 = centers.get(l, 0)[0];
			 double col2 = centers.get(l, 1)[0];
			 double col3 = centers.get(l, 2)[0];
			 Core.putText(bott, l+1+":", new Point(10+l*swidth,40), Core.FONT_HERSHEY_SIMPLEX, 1.5, new Scalar(255,255,255),2);
			 Core.rectangle(bott, new Point(58+l*swidth,5), new Point(rectwidth+l*swidth,45), new Scalar(255,255,255));
			 Core.rectangle(bott, new Point(59+l*swidth,6), new Point(rectwidth-1+l*swidth,44), new Scalar(col1,col2,col3),-1);
		 }
		Mat reInv2 = new Mat(new Size(reInv.cols(),reInv.rows()),CvType.CV_8UC3); 
		Imgproc.cvtColor(reInv, reInv2, Imgproc.COLOR_GRAY2BGR);
		System.out.println(reInv);
		System.out.println(reInv2);
		System.out.println(bott);
		reInv2.push_back(bott);
		Highgui.imwrite("PBN.png", reInv2);
		
		return reInv;
	}


}
