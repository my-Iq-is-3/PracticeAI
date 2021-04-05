package me.zach.anticheat.aicore.ai.firstai;

import me.zach.anticheat.aicore.AIType;
import me.zach.anticheat.aicore.SuperAI;

public class FirstAI extends SuperAI {
    public FirstAI(boolean isStatic){
        this.isStatic = isStatic;
    }

    @Override
    public AIType getType() {
        return AIType.FIRST_AI;
    }


    @Override
    public SuperAI getNew() {
        return new FirstAI(false);
    }

}
