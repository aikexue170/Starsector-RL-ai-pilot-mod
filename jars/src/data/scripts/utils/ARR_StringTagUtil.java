package data.scripts.utils;

import com.fs.starfarer.api.Global;

public class ARR_StringTagUtil {
    private static final String TAG_SYSTEMS = "systems";
    private static final String TAG_CONDITIONS = "conditions";
    private static final String TAG_INDUSTRIES = "industries";

    public static String getString(String category, String id) {
        return Global.getSettings().getString(category, id);
    }

    public static String getSystemsString(String id) {
        return getString(TAG_SYSTEMS, id);
    }

    public static String getConditionsString(String id) {
        return getString(TAG_CONDITIONS, id);
    }

    public static String getIndustriesString(String id) {
        return getString(TAG_INDUSTRIES, id);
    }
}