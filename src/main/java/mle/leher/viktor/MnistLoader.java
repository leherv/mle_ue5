package mle.leher.viktor;


import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.URL;

/**
 * Liest die Daten aus den MNIST Datenbank-Datein aus um sie NN-freundlich darzustellen
 * Zusätzlich gibt es
 */


public class MnistLoader {
    private static final String TRAINING_LABELS = "datasets/train-labels.idx1-ubyte";
    private static final String TRAINING_IMAGES = "datasets/train-images.idx3-ubyte";
    private static final String TEST_LABELS = "datasets/t10k-labels.idx1-ubyte";
    private static final String TEST_IMAGES = "datasets/t10k-images.idx3-ubyte";


    static double[][] getTrainingsImages() {
        try {
            return normalize(readImages(TRAINING_IMAGES));
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }

    }

    static double[][] getTrainingsLabels() {
        try {
            return readLabels(TRAINING_LABELS);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }

    }

    static double[][] getTestImages() {
        try {
            return normalize(readImages(TEST_IMAGES));
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }


    }

    static double[][] getTestLabels() {
        try {
            return readLabels(TEST_LABELS);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private static byte[][] readImages(String file) throws IOException {
        FileInputStream f = null;

        int magicNumber;        //0-3 byte
        int size;               //4-7 byte
        int rows;               //8-11
        int columns;            //12-15
        byte[][] images;         //jedes nächste byte ein pixel, jedes bild enhält rows*columns px, es gibt size viele bilder

        f = new FileInputStream(file);

        final byte[] integer = new byte[4];

        f.read(integer);
        magicNumber = byteArrayToInt(integer);

        if (magicNumber != 2051) {
            throw new IOException("Not a valid file");
        }

        f.read(integer);
        size = byteArrayToInt(integer);

        f.read(integer);
        rows = byteArrayToInt(integer);

        f.read(integer);
        columns = byteArrayToInt(integer);

        // 10000 bilder mit rows und colums geflatted (784 pixel nebeneinander)
        images = new byte[size][rows * columns];


        int nrImg = 0;

        while (nrImg < size) {

            byte[] image = new byte[rows * columns];
            f.read(image);
            images[nrImg] = image;
            if (images[nrImg][0] == -1) {
                return images;
            }
            nrImg++;
        }
        return images;
    }

    private static double[][] readLabels(String file) throws IOException {
        FileInputStream f = null;

        int magicNumber;
        int length;
        double[][] labels;
        byte[] integer = new byte[4];
        f = new FileInputStream(new File(file));

        f.read(integer);
        magicNumber = byteArrayToInt(integer);

        if (magicNumber != 2049) {
            throw new RuntimeException("Wrong File");
        }
        f.read(integer);
        length = byteArrayToInt(integer);

        labels = new double[length][10];
        for (int i = 0; i < labels.length; i++) {
            double[] label = new double[10];
            label[f.read() & 0xff] = 1;
            labels[i] = label;
        }
        return labels;

    }

    private static int byteArrayToInt(byte[] b) {
        if (b.length == 4)
            return b[0] << 24 | (b[1] & 0xff) << 16 | (b[2] & 0xff) << 8
                    | (b[3] & 0xff);
        else if (b.length == 2)
            return 0x00 << 24 | 0x00 << 16 | (b[0] & 0xff) << 8 | (b[1] & 0xff);

        return 0;
    }

    private static double[][] normalize(byte[][] img) {
        double[][] dImg = new double[img.length][img[0].length];
        for (int i = 0; i < img.length; i++) {
            for (int j = 0; j < img[i].length; j++) {
                // signed 2s complement example 11111111 10101100 (-84) wird 10101100 (172)
                dImg[i][j] = (double) ((int) img[i][j] & 0xFF) / 255;
            }
        }
        return dImg;
    }

    private double[] normalize(byte[] img) {
        double[] dImg = new double[img.length];
        for (int i = 0; i < img.length; i++) {
            // signed 2s complement example 11111111 10101100 (-84) wird 10101100 (172)
            dImg[i] = (double) ((int) img[i] & 0xFF) / 255;
        }
        return dImg;
    }

    private static String imageToString(double[] img) {
        String s = "";
        for (int i = 0; i < img.length; i++) {
            if (img[i] < .1) {
                s += " ";
            } else if (img[i] < .5) {
                s += ".";
            } else if (img[i] < .7) {
                s += "*";
            } else {
                s += "#";
            }
            if (i % 28 == 27) {
                s += "\n";
            }
        }
        return s;
    }

    public static void printDigit(int dig, double[][] img) {
        printDigit(img[dig]);
    }

    private static void printDigit(double[] img) {
        System.out.println(imageToString(img));
    }


//    private static URL getResource(String fileName) {
//        return MnistLoader.class.getClassLoader().getResource(fileName);
//    }
//
//    private static String getPathTo(String fileName) {
//        URL url = getResource(fileName);
//        if (url != null) {
//            return url.getPath();
//        }
//        return "";
//    }

}

