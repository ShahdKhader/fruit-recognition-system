module com.example.hw2ai {
    requires javafx.controls;
    requires javafx.fxml;


    opens com.example.hw2ai to javafx.fxml;
    exports com.example.hw2ai;
}