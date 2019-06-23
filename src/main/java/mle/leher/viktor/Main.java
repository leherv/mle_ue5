package mle.leher.viktor;

import com.opencsv.CSVWriter;
import org.javatuples.Pair;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Main {

    public static void main(String[] args) {
        writeCsv(classify(loadNeuralNet("neuralnets/neuralnet_0.009_3hidden.txt"), new TrainData(true)));
//        createNeuralNet(new int[]{784, 89, 29, 17, 10}, 0.001, 100, 0.05, 0.9);
    }


    private static void createNeuralNet(int[] layers, double threshold, int maxCycle, double learningRate, double momentum) {
        Network n = new Network(layers);
        var neuralTraining = new NeuralTraining(n, learningRate, momentum, new TrainData(false), new TrainData(true));
        neuralTraining.train(maxCycle, threshold);
        neuralTraining.exportWeights();
    }

    private static NeuralNet loadNeuralNet(String file) {
        return new NeuralNet(file);
    }

    private static Pair<int[], int[]> classify(NeuralNet neuralNet, TrainData testData) {
        return neuralNet.classify(testData);
    }

    private static void writeCsv(Pair<int[], int[]> classification) {
        try (Writer writer = Files.newBufferedWriter(Paths.get("output/classification.csv"));
             CSVWriter csvWriter = new CSVWriter(writer)
        ) {
            String[] headerRecord = {"predictedValues", "trueValues"};
            csvWriter.writeNext(headerRecord);

            for(int i = 0; i < classification.getValue0().length; i++) {
                String[] line = {classification.getValue0()[i] + "," + classification.getValue1()[i]};
                csvWriter.writeNext(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
