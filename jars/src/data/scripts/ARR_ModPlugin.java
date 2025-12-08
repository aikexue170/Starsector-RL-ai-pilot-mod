package data.scripts;

import com.fs.starfarer.api.BaseModPlugin;
import com.fs.starfarer.api.Global;
import com.fs.starfarer.api.impl.campaign.shared.SharedData;
import org.dark.shaders.util.ShaderLib;


public class ARR_ModPlugin extends BaseModPlugin {
    public static boolean hasGraphicsLib;

    @Override
    public void onGameLoad(boolean newGame) {

    }

    @Override
    public void beforeGameSave() {

    }

    @Override
    public void onApplicationLoad() {


        hasGraphicsLib = Global.getSettings().getModManager().isModEnabled("shaderLib");



        if (hasGraphicsLib) {

            ShaderLib.init();

            // LightData.readLightDataCSV("data/lights/ARR_light_data.csv");

            // TextureData.readTextureDataCSV("data/lights/ARR_texture_data.csv");

        }


    }

    @Override
    public void onNewGame() {

        if (NEX()) {
            new data.scripts.world.ARR_NEXGenerate().generate(Global.getSector());
        } else {
            new ARR_WorldGenerate().generate(Global.getSector());
        }
    }



    public static boolean NEX() {
        return Global.getSettings().getModManager().isModEnabled("nexerelin");
    }
}