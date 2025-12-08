package data.scripts.world;

import com.fs.starfarer.api.campaign.SectorAPI;
import data.scripts.ARR_WorldGenerate;
import exerelin.campaign.SectorManager;

public class ARR_NEXGenerate extends ARR_WorldGenerate {

    @Override
    public void generate(SectorAPI sector) {
        if (SectorManager.getManager().isCorvusMode()) { // 检测 nex 是否为原版地图模式
            super.generate(sector);
        }

    }
}