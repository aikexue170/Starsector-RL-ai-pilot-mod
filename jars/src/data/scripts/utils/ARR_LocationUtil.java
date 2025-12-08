package data.scripts.utils;

import com.fs.starfarer.api.combat.ShipAPI;
import org.lwjgl.util.vector.Vector2f;

public class ARR_LocationUtil {

    /**
     * 在舰船的相对坐标系下，向船首或船尾偏移。正为船首，负为船尾。
     * @param ship
     * @param offset
     * @return 偏移后的二维向量
     */
    public static Vector2f offsetPositionOnShip(ShipAPI ship, float offset){
        // 获取飞船的位置
        Vector2f shipLocation = new Vector2f(ship.getLocation());

        // 获取飞船的朝向角度（以度为单位）
        float facingAngle = ship.getFacing();
        // 将角度从度转换为弧度，因为 sin 和 cos 函数需要弧度作为参数
        float angleInRadians = (float) Math.toRadians(facingAngle);
        // 计算船尾方向的偏移量
        float offsetX = offset * (float) Math.cos(angleInRadians);
        float offsetY = offset * (float) Math.sin(angleInRadians);
        // 创建一个新的 Vector2f 对象，表示船尾方向的偏移
        Vector2f vectorOffset = new Vector2f(offsetX, offsetY);
        // 将偏移应用到飞船的位置上，得到新的生成位置
        Vector2f.add(shipLocation, vectorOffset, shipLocation);

        return shipLocation;
    }
}
