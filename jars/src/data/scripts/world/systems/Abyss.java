package data.scripts.world.systems;

import com.fs.starfarer.api.campaign.*;
import com.fs.starfarer.api.impl.campaign.procgen.NebulaEditor;
import com.fs.starfarer.api.impl.campaign.terrain.HyperspaceTerrainPlugin;
import com.fs.starfarer.api.util.Misc;

import static data.scripts.utils.ARR_StringTagUtil.getSystemsString;


// =====================================================================================================================
// >> START


public class Abyss {

    public void generate(SectorAPI sector) {
        StarSystemAPI system = sector.createStarSystem("Abyss");
        system.getLocation().set(7000, -10000);
        system.setBackgroundTextureFilename("graphics/ARR/backgrounds/Abyss_background.jpg");
        //system.setLightColor(new Color(200, 200, 210));
        PlanetAPI star = system.initStar("Abyss", "abyss_star", 400f, 350f, 3f, 0.75f, 1f);
        star.setName(getSystemsString("abyss_star"));
        star.setCustomDescriptionId("abyss_star");
        system.autogenerateHyperspaceJumpPoints(true, true);
        cleanup(system);
    }
    private void cleanup(StarSystemAPI system) {
        HyperspaceTerrainPlugin plugin = (HyperspaceTerrainPlugin) Misc.getHyperspaceTerrain().getPlugin();
        NebulaEditor editor = new NebulaEditor(plugin);
        float minRadius = plugin.getTileSize() * 2F;
        float radius = system.getMaxRadiusInHyperspace();
        editor.clearArc(system.getLocation().x, system.getLocation().y,
                0, radius + minRadius * 0.5F, 0, 360F);
        editor.clearArc(system.getLocation().x, system.getLocation().y,
                0, radius + minRadius, 0, 360F, 0.25F);
    }
}

// >> END
// =====================================================================================================================