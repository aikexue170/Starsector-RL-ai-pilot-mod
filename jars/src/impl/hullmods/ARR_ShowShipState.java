package impl.hullmods;

import com.fs.starfarer.api.Global;
import com.fs.starfarer.api.combat.BaseHullMod;
import com.fs.starfarer.api.combat.CombatEngineAPI;
import com.fs.starfarer.api.combat.ShipAPI;
import data.scripts.utils.ARR_EntityTimerManager;
import data.scripts.utils.ARR_Timer;
import org.lwjgl.util.vector.Vector2f;

import java.awt.*;

public class ARR_ShowShipState extends BaseHullMod {
    private static final ARR_EntityTimerManager timerManager = ARR_EntityTimerManager.getInstance();

    float[] lasers = new float[16];
    @Override
    public void advanceInCombat(ShipAPI ship, float amount){
        if (ship == null || !ship.isAlive()) return;
        CombatEngineAPI engine = Global.getCombatEngine();
        Vector2f loc = ship.getLocation();
        ARR_Timer timer = timerManager.getTimerForEntity(ship);

        if(timerManager.isTargetReachedForEntity(ship, engine, 2f)){
            timerManager.resetTimerForEntity(ship);
            engine.addFloatingText(new Vector2f(loc.x+200, loc.y+0),
                    "位置: " + loc.x + ", " + loc.y, 20f, Color.white, ship, 0.0001f, 0.0001f);
            engine.addFloatingText(new Vector2f(loc.x+200, loc.y+50),
                    "角度: " + ship.getFacing(), 20f, Color.red, ship, 0.0001f, 0.0001f);
            engine.addFloatingText(new Vector2f(loc.x+200, loc.y+100),
                    "速度: " + ship.getVelocity().x + ", " + ship.getVelocity().y, 20f, Color.GREEN, ship, 0.0001f, 0.0001f);
            engine.addFloatingText(new Vector2f(loc.x+200, loc.y+150),
                    "角速度: " + ship.getAngularVelocity(), 20f, Color.blue, ship, 0.0001f, 0.0001f);

            for (int key = 0; key < 16; key++) {
                Object val = engine.getCustomData().get("laser" + key);
                if (val instanceof Number) {          // 任何数字类型都兼容
                    lasers[key] = ((Number) val).floatValue();
                } else {
                    lasers[key] = 1.0f;               // 默认值，或按需要处理
                }
            }
            engine.addFloatingText(new Vector2f(loc.x, loc.y+400),
                    "laser传感器距离: " + lasers[0] + ", " + lasers[1] + ", " + lasers[2] + ", " + lasers[3] + ", " + lasers[4] + ", " + lasers[5] + ", " + lasers[6] + ", " + lasers[7] + ", " + lasers[8] + ", " + lasers[9] + ", " + lasers[10] + ", " + lasers[11] + ", " + lasers[12] + ", " + lasers[13] + ", " + lasers[14] + ", " + lasers[15],
                    20f, Color.cyan, ship, 0.0001f, 0.0001f);

        }

    }
}