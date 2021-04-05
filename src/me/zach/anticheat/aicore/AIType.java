package me.zach.anticheat.aicore;


import me.zach.anticheat.aicore.ai.firstai.FirstAI;

@SuppressWarnings("unused")
public enum AIType {

    FIRST_AI("first_ai",new FirstAI(true));

    private final String id;
    private final SuperAI ai;
    AIType(String identification, SuperAI ai){
        this.id = identification;
        this.ai = ai;
    }

    public String getId(){
        return id;
    }

    public SuperAI getAI(){
        return ai;
    }
}
