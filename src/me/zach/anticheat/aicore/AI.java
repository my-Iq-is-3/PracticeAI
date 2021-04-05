package me.zach.anticheat.aicore;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;

public class AI {
    AIType type;
    SuperAI ai;


    public AI(AIType type){
        this.type = type;
        ai = type.getAI().getNew();
        ai.init();
    }

    private AI(){};

    public SuperAI getAI(){
        return ai;
    }

}

