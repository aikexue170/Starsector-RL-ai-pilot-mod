package data.scripts.utils;

import com.fs.starfarer.api.combat.CombatEngineAPI;
import com.fs.starfarer.api.combat.CombatEntityAPI;

import java.util.HashMap;
import java.util.Map;

public class ARR_EntityTimerManager {

    // 单例模式：确保只有一个计时器管理器实例
    private static final ARR_EntityTimerManager instance = new ARR_EntityTimerManager();

    private final Map<CombatEntityAPI, ARR_Timer> entityTimers = new HashMap<>();

    // 私有构造函数，防止外部实例化
    private ARR_EntityTimerManager() {}

    // 获取单例实例
    public static ARR_EntityTimerManager getInstance() {
        return instance;
    }

    // 获取或创建一个实体的计时器
    public ARR_Timer getTimerForEntity(CombatEntityAPI entity) {
        synchronized (this) {
            if (!entityTimers.containsKey(entity)) {
                ARR_Timer newTimer = new ARR_Timer();
                entityTimers.put(entity, newTimer);
                return newTimer;
            } else {
                return entityTimers.get(entity);
            }
        }
    }

    // 重置特定实体的计时器
    public void resetTimerForEntity(CombatEntityAPI entity) {
        synchronized (this) {
            ARR_Timer timer = entityTimers.get(entity);
            if (timer != null) {
                timer.reset();
            }
        }
    }

    // 检查特定实体的计时器是否达到目标时间
    public boolean isTargetReachedForEntity(CombatEntityAPI entity, CombatEngineAPI engine, float timingTarget) {
        synchronized (this) {
            ARR_Timer timer = getTimerForEntity(entity);
            return timer.isTargetReached(engine, timingTarget);
        }
    }

    // 获取特定实体的计时器总时间
    public float getTotalTimeForEntity(CombatEntityAPI entity) {
        synchronized (this) {
            ARR_Timer timer = entityTimers.get(entity);
            return timer != null ? timer.getTotalTime() : 0f;
        }
    }

    // 暂停特定实体的计时器
    public void pauseTimerForEntity(CombatEntityAPI entity) {
        synchronized (this) {
            ARR_Timer timer = entityTimers.get(entity);
            if (timer != null) {
                timer.pause();
            }
        }
    }

    // 恢复特定实体的计时器
    public void resumeTimerForEntity(CombatEntityAPI entity, CombatEngineAPI engine) {
        synchronized (this) {
            ARR_Timer timer = entityTimers.get(entity);
            if (timer != null) {
                timer.resume(engine);
            }
        }
    }
}