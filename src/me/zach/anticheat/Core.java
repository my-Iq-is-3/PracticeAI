package me.zach.anticheat;

import me.zach.anticheat.aicore.AI;
import me.zach.anticheat.aicore.AIType;
import me.zach.anticheat.aicore.ai.firstai.AIUtil;
import org.neuroph.core.Layer;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

import java.time.Instant;
import java.util.Random;

public class Core {
    static Random r = new Random();
    public static void main(String[] args) throws InterruptedException {
        AI aitest = new AI(AIType.FIRST_AI);

        DataSet ds = new DataSet(4,1);
        addRows(ds,1000);

        Layer l = new Layer();
        for(int ig=0;ig<5;ig++){
            Neuron n = new Neuron();
            for(Object all: aitest.getAI().getNN().getInputNeurons()) {
                n.addInputConnection((Neuron) all);
            }
            l.addNeuron(n);
        }
        aitest.getAI().getNN().addLayer(l);

        Layer l1 = new Layer();
        for(int ig=0;ig<2;ig++){
            Neuron n = new Neuron();
            for(Object all: aitest.getAI().getNN().getLayerAt(0).getNeurons()) {
                n.addInputConnection((Neuron) all);
            }
            l1.addNeuron(n);
        }
        aitest.getAI().getNN().addLayer(l1);

        System.out.println("training...");
        long timebefore = Instant.now().getEpochSecond();
        aitest.getAI().train(ds);
        System.out.println("Finished training in " + (Instant.now().getEpochSecond()-timebefore) + " seconds.");

        DataSet ds1 = new DataSet(4,1);
        addRows(ds1,2000);
        aitest.getAI().test(ds1,true,89);
    }

    private static int r(){
        return r.nextInt(6)-2;
    }


    private static void addRows(DataSet ds,int amount){
        System.out.println("\nadding ds rows...");
        long timebefore = Instant.now().getEpochSecond();
        for(int i=0;i<amount;i++){

            double[] in = {r(),r(),r(),r()};
            double[] out = {1};
            for(double d : in){
                if(d != 1) {
                    out = new double[]{0};
                    break;
                }
            }
            if(r.nextInt(9) == 1){
                in = new double[]{1,1,1,1};
                out = new double[]{1};
            }
            ds.add(new DataSetRow(in,out));
        }
        System.out.println("added ds rows in " + (Instant.now().getEpochSecond()-timebefore) + " seconds\n");

    }
}
