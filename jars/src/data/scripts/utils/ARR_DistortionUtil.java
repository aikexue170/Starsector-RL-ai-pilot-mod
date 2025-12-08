package data.scripts.utils;

import org.dark.shaders.distortion.DistortionShader;
import org.dark.shaders.distortion.RippleDistortion;
import org.lwjgl.util.vector.Vector2f;

public class ARR_DistortionUtil
{
    public static void spawnDistortion(Vector2f location, Vector2f velocity, float size, float intensity, float frameRate, float fadeInSize, float fadeOutIntensity) {
        // 创建一个新的 RippleDistortion 对象，传入位置和速度
        RippleDistortion ripple = new RippleDistortion(location, velocity);
        // 设置扭曲效果的大小
        ripple.setSize(size);
        // 设置扭曲效果的强度
        ripple.setIntensity(intensity);
        // 设置扭曲效果的帧率
        ripple.setFrameRate(frameRate);
        // 设置扭曲效果的渐入大小
        ripple.fadeInSize(fadeInSize);
        // 设置扭曲效果的渐出强度
        ripple.fadeOutIntensity(fadeOutIntensity);
        // 将扭曲效果添加到全局的扭曲管理器中
        DistortionShader.addDistortion(ripple);
    }

    /**
     * 生成一个环状扭曲效果。
     *
     * @param location 涟漪的初始位置（通常是舰船的位置）
     * @param velocity 涟漪的移动速度（通常为零，因为环状扭曲是静态的）
     * @param radius 涟漪的最大半径
     * @param in 扩散时间（秒），涟漪从初始大小增加到最大半径所需的时间
     * @param dur 持续时间（秒），涟漪保持最大半径不变的时间
     * @param out 收缩时间（秒），涟漪从最大半径缩小回0所需的时间
     **/

    public static void ringDistortion(Vector2f location, Vector2f velocity,
                                      float radius, float in, float dur, float out) {
        // 创建一个新的 RippleDistortion 对象，传入位置和速度
        RippleDistortion ripple = new RippleDistortion(location, velocity);
        ripple.setCurrentFrame(30);
        // 设置最大大小为传入的半径
        ripple.setMaxSize(radius);
        // 设置初始强度为10.0f（可以根据需要调整）
        ripple.setIntensity(100.0f);
        // 设置最大强度为10.0f（可以根据需要调整）
        ripple.setMaxIntensity(100.0f);
        // 设置帧率为60.0f（可以根据需要调整）
        ripple.setFrameRate(60.0f);
        // 设置渐入大小，使涟漪在"in"秒内从0扩大到最大半径
        ripple.fadeInSize(in);
        //ripple.fadeInIntensity(in);
        // 设置渐出大小，使涟漪在"out"秒内从最大半径缩小回0
        ripple.setAutoFadeSizeTime(out);
        // 设置渐出强度，使涟漪在"out"秒内从最大强度减弱到0
        ripple.fadeOutIntensity(out);

        // 设置生命周期为总时间（扩散 + 持续 + 收缩）
        float totalLifetime = in + dur + out;
        ripple.setLifetime(totalLifetime);

        // 将扭曲效果添加到全局的扭曲管理器中
        DistortionShader.addDistortion(ripple);

    }
}

