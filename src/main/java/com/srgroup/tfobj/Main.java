package com.srgroup.tfobj;

import com.srgroup.tfobj.detectors.Classifier;
import com.srgroup.tfobj.detectors.TFObjectDetector;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

public class Main {

    public static void main(String[] args) {

        String modelFilePath = "ssd_mobilenet_v1_coco_11_06_2017\\frozen_inference_graph.pb";
        String labelMapFilePath = "labels.txt";
        String imageFilePath = "test.jpg";
        String outputImageFilePath = "out.jpg";


        try {
            Classifier classifier = TFObjectDetector.create(modelFilePath, labelMapFilePath);

            BufferedImage image = ImageIO.read(new File(imageFilePath));
            List<Classifier.Recognition> recognitionList = classifier.recognizeImage(image);

            for (Classifier.Recognition recognition : recognitionList) {
                System.out.println("Title " + recognition.getTitle() + " Score " + recognition.getConfidence());
            }

            saveAnnotatedImage(image, recognitionList, outputImageFilePath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void saveAnnotatedImage(BufferedImage image, List<Classifier.Recognition> recognitions, String outputImageFilePath) throws IOException {


        Graphics2D g2D = image.createGraphics();

        g2D.setColor (Color.BLACK);
        g2D.setStroke(new BasicStroke((image.getWidth() * 5) / 1000));

        // 15. Draw rectangle1.

        for (Classifier.Recognition recognition : recognitions) {

            Rectangle rectangle = new Rectangle(
                    (int) recognition.getLocation().getMinX(),
                    (int) recognition.getLocation().getMinY(),
                    (int) (recognition.getLocation().getWidth() + 0.5),
                    (int) (recognition.getLocation().getHeight() + 0.5));

            g2D.draw(rectangle);
        }

        ImageIO.write(image, "jpeg", new File(outputImageFilePath));
    }

}