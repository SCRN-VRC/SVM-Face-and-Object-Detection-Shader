#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine.UI;

[ExecuteInEditMode]
public class BakeSV : EditorWindow {

    string SavePath1 = "Assets/SVM Detector/SupportVectors.asset";
    public TextAsset source;
    public RenderTexture imgSource;
    public string imgText;

    [MenuItem("Tools/SCRN/Bake Support Vectors")]
    static void Init()
    {
        var window = GetWindowWithRect<BakeSV>(new Rect(0, 0, 400, 250));
        window.Show();
    }

    void OnGUI()
    {
        GUILayout.Label("Bake Support Vectors", EditorStyles.boldLabel);
        EditorGUILayout.BeginHorizontal();
        source = (TextAsset) EditorGUILayout.ObjectField("SVM Output (.xml):", source, typeof(TextAsset), false);
        EditorGUILayout.EndHorizontal();

        if (GUILayout.Button("Bake!")) {
           
            if (source == null)
                ShowNotification(new GUIContent("Select the .xml output file"));
            else
                OnGenerateTexture();
        }
    }

    void OnGenerateTexture()
    {
        //Support vector groups
        Regex rgSVs = new Regex("(?<=<support_vectors>)[\\s\\S]*(?=</support_vectors>)");

        Match mSVs = rgSVs.Match(source.text);
        string strSVs = mSVs.Groups[0].Value;

        if (strSVs.Length == 0) {
            ShowNotification(new GUIContent("Wrong file format"));
            return;
        }

        //Individual support vector
        Regex rgSV = new Regex("(?<=<_>)[\\s\\S]*?(?=</_>)");
        //Floats with exponents
        Regex rgFloats = new Regex("[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?");
        
        MatchCollection mSV = rgSV.Matches(strSVs);
        CaptureCollection sv = mSV[0].Captures;
        MatchCollection mFloats = rgFloats.Matches(sv[0].Value);

        Texture2D tex = new Texture2D(800, 800, TextureFormat.RFloat, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Point;
        tex.anisoLevel = 1;

        int i = 0;
        foreach (Match iSV in mSV) {
            sv = iSV.Captures;
            mFloats = rgFloats.Matches(sv[0].Value);

            int j = 0;

            foreach (Match iFloat in mFloats) {
                CaptureCollection fl = iFloat.Captures;
                float stf = float.Parse(fl[0].Value);
                tex.SetPixel(j % 40 + (i % 20) * 40, j / 40 + (i / 20) * 40,
                    new Color(stf, 0.0f, 0.0f, 0.0f));
                j++;
            }
            i++;
        }

        Regex rgAlpha = new Regex("(?<=<alpha>)[\\s\\S]*(?=</alpha>)");
        Regex rgIndex = new Regex("(?<=<index>)[\\s\\S]*(?=</index>)");

        Match mAlpha = rgAlpha.Match(source.text);
        string strAlpha = mAlpha.Groups[0].Value;
        Match mIndex = rgIndex.Match(source.text);
        string strIndex = mIndex.Groups[0].Value;

        if (strAlpha.Length == 0 || strIndex.Length == 0) {
            ShowNotification(new GUIContent("Wrong file format"));
            return;
        }

        MatchCollection alphaF = rgFloats.Matches(strAlpha);
        MatchCollection indexF = rgFloats.Matches(strIndex);

        for (i = 0; i < mSV.Count; i++) {
            float alpha = float.Parse(alphaF[i].Captures[0].Value);
            float index = float.Parse(indexF[i].Captures[0].Value);
            tex.SetPixel(i, 798, new Color(index, 0.0f, 0.0f, 0.0f));
            tex.SetPixel(i, 799, new Color(alpha, 0.0f, 0.0f, 0.0f));
        }

        Regex rgRho = new Regex("(?<=<rho>)[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?");
        Match mRho = rgRho.Match(source.text);
        string strRho = mRho.Groups[0].Value;

        Regex rgGamma = new Regex("(?<=<gamma>)[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?");
        Match mGamma = rgGamma.Match(source.text);
        string strGamma = mGamma.Groups[0].Value;

        tex.SetPixel(799, 799, new Color(float.Parse(strRho), 0.0f, 0.0f, 0.0f));
        tex.SetPixel(798, 799, new Color(float.Parse(strGamma), 0.0f, 0.0f, 0.0f));
        tex.SetPixel(797, 799, new Color(mSV.Count, 0.0f, 0.0f, 0.0f));

        AssetDatabase.CreateAsset(tex, SavePath1);
        AssetDatabase.SaveAssets();

        ShowNotification(new GUIContent("Done"));
    }
}
#endif