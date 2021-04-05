package me.zach.anticheat;

import me.zach.anticheat.aicore.AI;
import me.zach.anticheat.aicore.AIType;
import org.neuroph.core.Layer;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.nnet.comp.layer.InputLayer;

public class Core {
    public static void main(String[] args) {
        AI aitest = new AI(AIType.FIRST_AI);
        DataSet ds = new DataSet(4,1);
        ds.add(new double[]{0,0,0,0},new double[]{0});
        ds.add(new double[]{0,0,0,1},new double[]{0});
        ds.add(new double[]{0,0,1,0},new double[]{0});
        ds.add(new double[]{1,1,0,0},new double[]{0});
        ds.add(new double[]{1,0.1,1,3},new double[]{0});
        ds.add(new double[]{1,1,1,1},new double[]{1});
        ds.add(new double[]{1,1,1,1},new double[]{1});
        ds.add(new double[]{0,1,2,5},new double[]{0});

        Layer l = new Layer();
        InputLayer il = new InputLayer(4);
        aitest.getAI().getNN().setInputNeurons(il.getNeurons());
        for(int ignored :new int[]{1,2,3,4}){
            Neuron n = new Neuron();
            for(Object all: aitest.getAI().getNN().getInputNeurons()) {
                n.addInputConnection((Neuron) all);
            }
            l.addNeuron(n);
        }

        aitest.getAI().getNN().addLayer(l);
        aitest.getAI().train(ds);
//        DataSet dseval = new DataSet(4);
//        dseval.add(new DataSetRow(new double[]{1,1,1,1},new double[]{1}));
//
//        AIUtil.testNeuralNetwork(aitest.getAI().getNN(),dseval);

    }
}
