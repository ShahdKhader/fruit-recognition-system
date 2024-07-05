package com.example.hw2ai;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.File;

public class HelloController {
    @FXML
    private ImageView apple, banana, orange;

    @FXML
    private Button buildNetworkButton, testButton, trainButton;
    @FXML
    private ComboBox<?> hiddenFunctions, outputFunctions;
    @FXML
    private TextField  epochsField, goalField, learningRateField, neuronsField, sweet, color;

    @FXML
    private Label testResult;
    @FXML
    private TextArea outputArea;

    @FXML
    private Button chooseFile;
    @FXML
    private FruitRecognitionSystem fruitRecognitionSystem ;
    @FXML
    private ActionEvent event;
    @FXML
    void chooseFileClicked(ActionEvent event) {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Choose Training Data File");
        fileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("Text Files", "*.txt"),
                new FileChooser.ExtensionFilter("All Files", ".")
        );
        File selectedFile = fileChooser.showOpenDialog(new Stage());
        if (selectedFile != null) {
            String filePath = selectedFile.getAbsolutePath();
            System.out.println("Selected File: " + filePath);

            fruitRecognitionSystem.prepareTrainingData(filePath);

            outputArea.appendText("Training data loaded from: " + filePath + "\n");
        } else {
            System.out.println("No file selected.");
        }
    }
    @FXML
    private void buildNeuralNetwork() {
        int inputNeurons = 2; // Sweetness and color
        int hiddenNeurons = Integer.parseInt(neuronsField.getText());
        int outputNeurons = 3; // Three classes: Apple, Banana, Orange

        fruitRecognitionSystem = new FruitRecognitionSystem();
        fruitRecognitionSystem.buildNeuralNetwork(inputNeurons, hiddenNeurons, outputNeurons);

        outputArea.appendText("Neural Network built with " + inputNeurons + " input neurons, "
                + hiddenNeurons + " hidden neurons, and " + outputNeurons + " output neurons.\n");
        System.out.println("Neural Network built with " + inputNeurons + " input neurons, "
                + hiddenNeurons + " hidden neurons, and " + outputNeurons + " output neurons.\n");
    }

    @FXML
    private void trainNeuralNetwork() {
        if (fruitRecognitionSystem == null) {
            outputArea.appendText("Error: Build the neural network first.\n");
            return;
        }

        double learningRate = Double.parseDouble(learningRateField.getText());
        int epochs = Integer.parseInt(epochsField.getText());
        double goal = Double.parseDouble(goalField.getText());
        fruitRecognitionSystem.prepareTrainingData("fruit.txt");
        String hidden= hiddenFunctionsSelected(event);
        String output= outputFunctionsSelected(event);
        fruitRecognitionSystem.trainNeuralNetwork(learningRate, epochs, goal, hidden, output);

        outputArea.appendText("Neural Network trained with learning rate " + learningRate
                + ", " + epochs + " epochs, and goal " + goal + ".\n");
    }
    @FXML
    private void testNeuralNetwork() {
        String hidden= hiddenFunctionsSelected(event);
        String output= hiddenFunctionsSelected(event);
        if (fruitRecognitionSystem == null) {
            outputArea.appendText("Error: Build and train the neural network first.\n");
            return;
        }

        double sweetness = Double.parseDouble(sweet.getText());
        double color1 = Double.parseDouble(color.getText());

        String expectedClass = getExpectedClass();
        String predictedClass = fruitRecognitionSystem.testNeuralNetwork(sweetness, color1, hidden,output);
        testResult.setText(predictedClass);
        outputArea.appendText("Test Result: Expected = " + expectedClass + ", Predicted = " + predictedClass + "\n");
    }

    private String getExpectedClass() {
        double sweetness = Double.parseDouble(sweet.getText());
        double color1 = Double.parseDouble(color.getText());

        if (sweetness > 6.0 && color1 < 0.6) {
            return "Apple";
        } else if (sweetness < 3.0 && color1 > 0.3) {
            return "Orange";
        } else {
            return "Banana";
        }
    }

    @FXML
    public String hiddenFunctionsSelected(ActionEvent event) {
        String selectedHiddenFunction = (String) hiddenFunctions.getValue();
        if(selectedHiddenFunction==null){
            selectedHiddenFunction="sigmoid";
        }
        System.out.println("Selected Hidden Function: " + selectedHiddenFunction);
        return selectedHiddenFunction;
    }


    @FXML
    private String outputFunctionsSelected(ActionEvent event) {
        String selectedOutputFunction = (String) outputFunctions.getValue();
        if (selectedOutputFunction == null) {
            selectedOutputFunction = "sigmoid";
        }
        System.out.println("Selected Output Function: " + selectedOutputFunction);
        return selectedOutputFunction;
    }
    @FXML
    void testClicked(ActionEvent event) {
        String hidden= hiddenFunctionsSelected(event);
        String output= hiddenFunctionsSelected(event);
        double sweetness = Double.parseDouble(sweet.getText());
        double color1 = Double.parseDouble(color.getText());
        String predictedClass = fruitRecognitionSystem.testNeuralNetwork(sweetness, color1, hidden,output);
        if ("Apple".equals(predictedClass)) {
            apple.setVisible(true);
            orange.setVisible(false);
            banana.setVisible(false);
        } else if ("Orange".equals(predictedClass)) {
            apple.setVisible(false);
            orange.setVisible(true);
            banana.setVisible(false);
        } else {
            apple.setVisible(false);
            orange.setVisible(false);
            banana.setVisible(true);
        }

    }

    String hiddenFunctionsString (){
        return  hiddenFunctionsSelected(event);
    }
    String outputFunctionsString(){
        return  outputFunctionsSelected(event);
    }

}