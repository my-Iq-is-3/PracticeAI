package me.zach.anticheat.aicore.ai.firstai;

import me.zach.anticheat.aicore.SuperAI;
import org.neuroph.core.Connection;
import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.learning.error.MeanSquaredError;
import org.neuroph.eval.ClassifierEvaluator;
import org.neuroph.eval.ErrorEvaluator;
import org.neuroph.eval.Evaluation;
import org.neuroph.eval.classification.ClassificationMetrics;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.nnet.learning.BackPropagation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

public class AIUtil {

    /**
     * Prints network output for the each element from the specified training
     * set.
     *
     * @param neuralNet neural network
     * @param testSet test data set
     */
    public static void testNeuralNetwork(NeuralNetwork neuralNet, org.neuroph.core.data.DataSet testSet,boolean retest,double threshold,long iteration,double highAcc) throws InterruptedException {
        try {
            double correct = 0;
            double hAcc = highAcc;
            double inc = 0;
            ArrayList<DataSetRow> failed = new ArrayList<>();
            System.out.println("\nTesting...");

            for (DataSetRow testSetRow : testSet.getRows()) {
                neuralNet.setInput(testSetRow.getInput());
                neuralNet.calculate();
                double[] networkOutput = neuralNet.getOutput();

                if (networkOutput[0] == testSetRow.getDesiredOutput()[0]) correct++;
                else {
                    inc++;
                    failed.add(testSetRow);
                }
                if (testSet.getRows().size() < 6) {
                    System.out.println("Input: " + Arrays.toString(testSetRow.getInput()));
                    System.out.println("Output: " + Arrays.toString(networkOutput));
                    System.out.println("Desired output" + Arrays.toString(testSetRow.getDesiredOutput()));
                }
            }
            double total = inc + correct;
            double acc = (correct / total) * 100;
            System.out.println("testing finished, results: \nCorrect: " + correct + "\nIncorrect: " + inc + "\nPercent correct: " + acc + "%");
            if(acc>hAcc) hAcc = acc;
            if (!retest) return;
            if (acc > threshold) {
                System.out.println("Threshold reached! (" + threshold + "%)");
                return;
            }
            if(iteration > 3000){
                System.out.println("Max iteration reached, highest accuracy: " + hAcc + "%");
                return;
            }
            System.out.println("Re-training... (" + iteration + ")");
            for (DataSetRow tsr : failed) {
                DataSet relearn = new DataSet(4, 1);
                relearn.add(tsr);
                neuralNet.learn(relearn);
            }
            Thread.sleep(10);
            AIUtil.testNeuralNetwork(neuralNet, testSet, true, threshold, iteration + 1,hAcc);
        }catch (StackOverflowError | AssertionError e){
            System.out.println("\nstack error -_- at iteration " + iteration);
            return;
        }
    }

    public static void testNeuralNetwork(NeuralNetwork neuralNet, org.neuroph.core.data.DataSet testSet) throws InterruptedException {
        testNeuralNetwork(neuralNet,testSet,false,0,0,0);
    }

    public static void testNeuralNetwork(NeuralNetwork neuralNet, org.neuroph.core.data.DataSet testSet,boolean retest,double threshold) throws InterruptedException {
        testNeuralNetwork(neuralNet,testSet,retest,threshold,0,0);
    }


    public static void evaluate(NeuralNetwork neuralNet, DataSet dataSet) {

        System.out.println("Calculating performance indicators for neural network.");

        Evaluation evaluation = new Evaluation();
        evaluation.addEvaluator(new ErrorEvaluator(new MeanSquaredError()));

        String[] classLabels = new String[]{"1", "2", "3"};
        evaluation.addEvaluator(new ClassifierEvaluator.MultiClass(classLabels));
        evaluation.evaluate(neuralNet, dataSet);

        ClassifierEvaluator evaluator = evaluation.getEvaluator(ClassifierEvaluator.MultiClass.class);
        ConfusionMatrix confusionMatrix = evaluator.getResult();
        System.out.println("Confusion matrrix:\r\n");
        System.out.println(confusionMatrix.toString() + "\r\n\r\n");
        System.out.println("Classification metrics\r\n");
        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(confusionMatrix);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metrics);
        for (ClassificationMetrics cm : metrics) {
            System.out.println(cm.toString() + "\r\n");
        }
        System.out.println(average.toString());
    }

}
