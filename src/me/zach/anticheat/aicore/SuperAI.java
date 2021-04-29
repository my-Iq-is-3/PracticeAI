package me.zach.anticheat.aicore;

import me.zach.anticheat.aicore.ai.firstai.AIUtil;
import org.neuroph.core.Connection;
import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Perceptron;
import org.neuroph.nnet.learning.BackPropagation;

import java.util.Iterator;

public abstract class SuperAI {
    public boolean isStatic = true;
    public boolean init = false;
    protected NeuralNetwork fnn;
    public SuperAI(boolean isStatic){
        this.isStatic = isStatic;
    }

    protected SuperAI(){}


    public abstract AIType getType();


    public void init() {
        if(init) throw new IllegalStateException("cannot initialize an already initialized instance");
        if(isStatic) throw new IllegalStateException("cannot initialize a static instance, use #getNew");
        fnn = new Perceptron(4,1);

    }


    public void train(DataSet ds) {
        for(DataSetRow dsr : ds.getRows()){
            DataSet dsnew = new DataSet(ds.getInputSize(),ds.getOutputSize());
            dsnew.add(dsr);
            fnn.learn(dsnew);
        }
    }


    public NeuralNetwork getNN(){
        return fnn;
    }


    public void setNN(NeuralNetwork nn) {
        this.fnn = nn;
    }


    public void test(DataSet ds,boolean retest,double threshold) throws InterruptedException {
        AIUtil.testNeuralNetwork(this.fnn,ds,retest,threshold);
    }



    public abstract SuperAI getNew();
}
