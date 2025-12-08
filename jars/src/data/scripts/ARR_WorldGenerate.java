package data.scripts;

import com.fs.starfarer.api.campaign.FactionAPI;
import com.fs.starfarer.api.campaign.RepLevel;
import com.fs.starfarer.api.campaign.SectorAPI;
import com.fs.starfarer.api.campaign.SectorGeneratorPlugin;
import data.scripts.world.systems.Abyss;

// =====================================================================================================================
// >> START

/**
 * 定义UCSE_WorldGenerate类，接入SectorGeneratorPlugin接口。
 * 该类进行各星系的总体生成设定。
 * SectorGeneratorPlugin接口中只有一条抽象方法(generate()方法)，实现即可。
 */
public class ARR_WorldGenerate implements SectorGeneratorPlugin {
    @Override
    public void generate(SectorAPI sector) {
        // 在此准备生成星系
        new Abyss().generate(sector);
    }
}

// >> END
// =====================================================================================================================