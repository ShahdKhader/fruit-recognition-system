package com.example.hw2ai;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Scanner;

public class FruitRecognitionSystem {
    private double[][] weightsInputHidden;
    private double[][] weightsHiddenOutput;
    private double[][][] trainingSet;

    public void buildNeuralNetwork(int inputNeurons, int hiddenNeurons, int outputNeurons) {
        weightsInputHidden = initializeWeights(inputNeurons, hiddenNeurons);
        weightsHiddenOutput = initializeWeights(hiddenNeurons, outputNeurons);
    }

    private double[][] initializeWeights(int rows, int cols) {
        double[][] weights = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                weights[i][j] = Math.random() * 2.4 - 1.2;
            }
        }
        return weights;
    }

    public void prepareTrainingData(String fileName) {//s
        try {
            File file = new File(fileName);
            Scanner scanner = new Scanner(file);

            int numRows = 10;
            int numColumns = 2;
            trainingSet = new double[numRows][numColumns][];

            int rowIndex = 0;
            while (scanner.hasNextLine() && rowIndex < numRows) {
                String line = scanner.nextLine();
                String[] values = line.split(",");

                if (values.length >= numColumns) {
                    double[] input = new double[numColumns];
                    for (int i = 0; i < numColumns; i++) {
                        input[i] = Double.parseDouble(values[i]);
                    }

                    String expectedClass = getExpectedClass(input[0], input[1]);
                    double[] target = convertClassToTarget(expectedClass);

                    trainingSet[rowIndex][0] = input;
                    trainingSet[rowIndex][1] = target;
                    rowIndex++;
                } else {
                    System.out.println("Skipping invalid line: " + line);
                }
            }

            scanner.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        printTrainingSet();
    }

    public void printTrainingSet() {
        if (trainingSet != null) {
            for (int i = 0; i < trainingSet.length; i++) {
                System.out.println("Example " + (i + 1) + ":");
                System.out.println("Input: " + Arrays.toString(trainingSet[i][0]));
                System.out.println("Target: " + Arrays.toString(trainingSet[i][1]));
                System.out.println();
            }
        } else {
            System.out.println("Training set is not initialized.");
        }
    }

    private double[] convertClassToTarget(String expectedClass) {
        switch (expectedClass) {
            case "Apple":
                return new double[]{1, 0, 0};
            case "Banana":
                return new double[]{0, 1, 0};
            case "Orange":
                return new double[]{0, 0, 1};
            default:
                return new double[]{0, 0, 0};
        }
    }

    private String getExpectedClass(double sweetness, double color) {//Yd
        if (sweetness > 6.0 && color > 0.6) {
            return "Apple";
        } else if (sweetness < 3.0 && color > 0.3) {
            return "Orange";
        } else {
            return "Banana";
        }
    }

    public void trainNeuralNetwork(double learningRate, int epochs, double goal, String hidden, String output) {
        for (int epoch = 1; epoch <= epochs; epoch++) {
            double totalError = 0.0;
            int correctCount = 0;
            for (int i = 0; i < trainingSet.length; i++) {
                double[] input = trainingSet[i][0];
                double[] target = trainingSet[i][1];
                double[] hiddenOutput = calculateLayerHidden(input, weightsInputHidden, hidden);
                double[] networkOutput = calculateLayerOutput(hiddenOutput, weightsHiddenOutput, output);

                int predictedClass = getMaxIndex(networkOutput);
                switch (predictedClass) {
                    case 0:
                        System.out.println("Apple");
                        break;
                    case 1:
                        System.out.println("Banana");
                        break;
                    case 2:
                        System.out.println("Orange");
                        break;
                    default:
                        System.out.print("Unknown");
                }
                System.out.println("--------------------------------------------------");

                double[] outputError = calculateError(target, networkOutput);
                double[] hiddenError = calculateHiddenError(outputError, weightsHiddenOutput, hiddenOutput, hidden);

                double[] outputErrorDerivative = calculateLayerOutputDir(hiddenOutput, weightsHiddenOutput, output);
                double[] hiddenErrorDerivative = calculateLayerHiddenDir(input, weightsInputHidden, hidden);

                for (int j = 0; j < outputError.length; j++) {
                    outputError[j] *= outputErrorDerivative[j];
                }

                for (int j = 0; j < hiddenError.length; j++) {
                    hiddenError[j] *= hiddenErrorDerivative[j];
                }

                totalError += calculateTotalError(outputError);
                updateWeights(learningRate, outputError, hiddenOutput, weightsHiddenOutput);
                updateWeights(learningRate, hiddenError, input, weightsInputHidden);
                if (predictedClass == getMaxIndex(target)) {
                    correctCount++;
                }
            }
            double accuracy = (double) correctCount / trainingSet.length;
            System.out.println("Epoch #" + epoch + " Total Error: " + totalError + " Accuracy: " + accuracy);

            if (totalError < goal) {
                break;
            }
        }
    }//l

    private double[] calculateLayerOutputDir(double[] input, double[][] weights, String output) {
        double[] outputt;

        int numHiddenNeurons = weights[0].length; // Get the number of hidden neurons

        if (output.equalsIgnoreCase("sigmoid")) {
            outputt = new double[numHiddenNeurons];
            for (int j = 0; j < numHiddenNeurons; j++) {
                double sum = 0.0;
                for (int i = 0; i < weights.length; i++) {
                    sum += input[i] * weights[i][j];
                }
                outputt[j] = sigmoidDerivative(sum - 1);
            }
        } else if (output.equalsIgnoreCase("tanH")) {
            outputt = new double[numHiddenNeurons];
            for (int j = 0; j < numHiddenNeurons; j++) {
                double sum = 0.0;
                for (int i = 0; i < weights.length; i++) {
                    sum += input[i] * weights[i][j];
                }
                outputt[j] = tanhDerivative(sum - 1);
            }
        } else if (output.equalsIgnoreCase("softmax")) {
            outputt = softmaxDerivative(input, weights);
        } else {
            throw new IllegalArgumentException("Invalid output function: " + output);
        }

        return outputt;
    }

    private double[] calculateLayerHiddenDir(double[] input, double[][] weights, String hidden) {
        double[] hiddenDerivative = new double[weights[0].length];
        for (int j = 0; j < weights[0].length; j++) {
            double sum = 0.0;
            for (int i = 0; i < weights.length; i++) {
                sum += input[i] * weights[i][j];
            }
            if (hidden.equalsIgnoreCase("relu")) hiddenDerivative[j] = leakyReLUDerivative(sum - 1);
            else if (hidden.equalsIgnoreCase("tanH")) hiddenDerivative[j] = tanhDerivative(sum - 1);
        }
        return hiddenDerivative;
    }

    public String testNeuralNetwork(double sweetness, double color, String hidden, String output) {
        double[] input = {sweetness, color};
        double[] hiddenOutput = calculateLayerHidden(input, weightsInputHidden, hidden);
        double[] networkOutput = calculateLayerOutput(hiddenOutput, weightsHiddenOutput, output);

        int predictedClass = getMaxIndex(networkOutput);
        switch (predictedClass) {
            case 0:
                return "Apple";
            case 1:
                return "Banana";
            case 2:
                return "Orange";
            default:
                return "Unknown";
        }
    }

    private double[] calculateLayerOutput(double[] input, double[][] weights, String output) {
        double[] outputt;

        int numHiddenNeurons = weights[0].length; // Get the number of hidden neurons

        if (output.equalsIgnoreCase("sigmoid")) {
            outputt = new double[numHiddenNeurons];
            for (int j = 0; j < numHiddenNeurons; j++) {
                double sum = 0.0;
                for (int i = 0; i < weights.length; i++) {
                    sum += input[i] * weights[i][j];
                }
                outputt[j] = sigmoid(sum - 1);
            }
        } else if (output.equalsIgnoreCase("tanH")) {
            outputt = new double[numHiddenNeurons];
            for (int j = 0; j < numHiddenNeurons; j++) {
                double sum = 0.0;
                for (int i = 0; i < weights.length; i++) {
                    sum += input[i] * weights[i][j];
                }
                outputt[j] = tanh(sum - 1);
            }
        } else if (output.equalsIgnoreCase("softmax")) {
            outputt = softmax(input, weights);
        } else {
            throw new IllegalArgumentException("Invalid output function: " + output);
        }

        return outputt;
    }

    private double[] softmax(double[] input, double[][] weights) {
        double[] expValues = new double[weights[0].length];
        double sumExp = 0.0;

        for (int j = 0; j < weights[0].length; j++) {
            double sum = 0.0;
            for (int i = 0; i < weights.length; i++) {
                sum += input[i] * weights[i][j];
            }
            expValues[j] = Math.exp(sum);
            sumExp += expValues[j];
        }

        for (int j = 0; j < weights[0].length; j++) {
            expValues[j] /= sumExp;
        }

        return expValues;
    }

    public double[] softmaxDerivative(double[] input, double[][] weights) {
        double[] softmaxValues = softmax(input, weights);
        double[] derivative = new double[softmaxValues.length];

        for (int i = 0; i < softmaxValues.length; i++) {
            derivative[i] = softmaxValues[i] * (1 - softmaxValues[i]);
        }

        return derivative;
    }

    private double[] calculateLayerHidden(double[] input, double[][] weights, String hidden) {
        double[] output = new double[weights[0].length];
        for (int j = 0; j < weights[0].length; j++) {
            double sum = 0.0;
            for (int i = 0; i < weights.length; i++) {
                sum += input[i] * weights[i][j];
            }
            if (hidden.equalsIgnoreCase("relu")) output[j] = LeakyReLU(sum - 1);
            else if (hidden.equalsIgnoreCase("tanH")) output[j] = tanh(sum - 1);
        }
        return output;
    }

    private double LeakyReLU(double x) {
        return (x > 0) ? x : 0.01 * x;
    }

    private double leakyReLUDerivative(double x) {
        return (x > 0) ? 1 : 0.01;
    }

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x) {
        double sigmoidX = sigmoid(x);
        return sigmoidX * (1 - sigmoidX);
    }

    public static double tanh(double x) {
        return Math.tanh(x);
    }

    public static double tanhDerivative(double x) {
        double tanhX = tanh(x);
        return 1 - tanhX * tanhX;
    }



    private double[] calculateHiddenError(double[] outputError, double[][] weights, double[] hiddenOutput, String hidden) {
        double[] hiddenError = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            double weightedErrorSum = 0.0;
            for (int j = 0; j < outputError.length; j++) {
                weightedErrorSum += weights[i][j] * outputError[j];
            }
            if (hidden.equalsIgnoreCase("relu")) hiddenError[i] = leakyReLUDerivative(hiddenOutput[i]) * weightedErrorSum;
            else if (hidden.equalsIgnoreCase("tanH")) hiddenError[i] = tanhDerivative(hiddenOutput[i]) * weightedErrorSum;
        }
        return hiddenError;
    }

    private double[] calculateError(double[] target, double[] output) {
        if (target.length != output.length) {
            throw new IllegalArgumentException("Target and output arrays must have the same length");
        }

        double[] error = new double[target.length];
        for (int i = 0; i < target.length; i++) {
            error[i] = target[i] - output[i];
        }
        return error;
    }


    private double calculateTotalError(double[] error) {
        double sum = 0.0;
        for (double e : error) {
            sum += 0.5 * Math.pow(e, 2);
        }
        return sum;
    }

    private void updateWeights(double learningRate, double[] gradiantError, double[] input, double[][] weights) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] += learningRate * gradiantError[j] * input[i];
            }
        }
    }

    private int getMaxIndex(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }



}