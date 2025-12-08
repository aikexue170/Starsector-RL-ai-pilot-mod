package data.scripts.weapons;

import com.fs.starfarer.api.combat.*;
import org.lwjgl.util.vector.Vector2f;

public class ScanBeamEffect implements BeamEffectPlugin {

	// 射线最大距离，用于归一化
	private static final float MAX_BEAM_RANGE = 2000f;

	public void advance(float amount, CombatEngineAPI engine, BeamAPI beam) {
		// 获取光束的起点和终点
		Vector2f from = beam.getFrom();
		Vector2f to = beam.getRayEndPrevFrame();

		// 默认距··
		float distance = MAX_BEAM_RANGE;

		// 检查是否击中了需要避让的目标
		CombatEntityAPI target = beam.getDamageTarget();
		if(target instanceof ShipAPI){
			ShipAPI ship = (ShipAPI) target;
			// 只避让战列舰和巡洋舰
			if(ship.isCapital() || ship.isCruiser()){
				// 计算实际击中距离
				distance = Vector2f.sub(to, from, null).length();
			}
		}

		// 归一化距离到[0,1]范围
		float normalizedDistance = Math.min(distance / MAX_BEAM_RANGE, 1.0f);

		// 获取武器槽ID
		String slotID = beam.getWeapon().getSlot().getId();

		// 存储到engine的custom data中
		engine.getCustomData().put(slotID, normalizedDistance);

	}
}