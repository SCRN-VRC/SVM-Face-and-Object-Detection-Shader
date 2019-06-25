#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Text.RegularExpressions;

[ExecuteInEditMode]
public class BakeSV : EditorWindow {

    string SavePath1 = "Assets/SVM Face Detection/Textures/SVM-SupportVecs.asset";
    string SavePath2 = "Assets/SVM Face Detection/Textures/SVM-AlphasIndex.asset";
    string SavePath3 = "Assets/SVM Face Detection/Textures/LookupTable.asset";
    public TextAsset source;

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
        source = (TextAsset) EditorGUILayout.ObjectField("SVM Output (.yaml):", source, typeof(TextAsset));
        EditorGUILayout.EndHorizontal();

        if (GUILayout.Button("Bake!")) {
           
            if (source == null)
                ShowNotification(new GUIContent("Select the .yaml output file"));
            else
                OnGenerateTexture();
        }

        GUILayout.Label("Dev Tools", EditorStyles.boldLabel);
        if (GUILayout.Button("Generate 3D Index Lookup Table")) {
            OnGenerateLookUpTable();
        }
    }

    void OnGenerateLookUpTable() {
        Texture2D lookup = new Texture2D(1568, 1, TextureFormat.RGBAFloat, false);
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                for (int k = 0; k < 8; k++) {
                    int ind1 = ((i * 2) * 14 + (j * 2)) * 8 + k;
                    int ind2 = ((i * 2 + 1) * 14 + (j * 2)) * 8 + k;
                    int ind3 = ((i * 2) * 14 + (j * 2 + 1)) * 8 + k;
                    int ind4 = ((i * 2 + 1) * 14 + (j * 2 + 1)) * 8 + k;
                    float fi = (float) i;
                    float fj = (float) j;
                    float fk = (float) k;
                    lookup.SetPixel(ind1, 0, new Color(fi, fj, fk, 0.0f));
                    lookup.SetPixel(ind2, 0, new Color(fi, fj, fk, 2.0f));
                    lookup.SetPixel(ind3, 0, new Color(fi, fj, fk, 1.0f));
                    lookup.SetPixel(ind4, 0, new Color(fi, fj, fk, 3.0f));
                }
            }
        }
        AssetDatabase.CreateAsset(lookup, SavePath3);
        AssetDatabase.SaveAssets();
    }

    void OnGenerateTexture()
    {
        //Support vector groups
        Regex rgSVs = new Regex("(?<=support_vectors:)[\\s\\S]*(?=decision_functions:)");

        Match mSVs = rgSVs.Match(source.text);
        string strSVs = mSVs.Groups[0].Value;

        if (strSVs.Length == 0) {
            ShowNotification(new GUIContent("Wrong file format"));
            return;
        }

        //Individual support vector
        Regex rgSV = new Regex("(\\[[\\s\\S]*?\\])");
        //Floats with exponents
        Regex rgFloats = new Regex("[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?");
        
        MatchCollection mSV = rgSV.Matches(strSVs);
        CaptureCollection sv = mSV[0].Captures;
        MatchCollection mFloats = rgFloats.Matches(sv[0].Value);

        int rows = mSV.Count;
        int cols = mFloats.Count;

        Texture2D tex = new Texture2D(cols, rows, TextureFormat.RFloat, false);

        int i = 0;
        foreach (Match iSV in mSV) {
            sv = iSV.Captures;
            mFloats = rgFloats.Matches(sv[0].Value);

            int j = 0;

            foreach (Match iFloat in mFloats) {
                CaptureCollection fl = iFloat.Captures;
                float stf = float.Parse(fl[0].Value);
                tex.SetPixel(j, i, new Color(stf, 0.0f, 0.0f, 0.0f));
                j++;
            }
            i++;
        }

        AssetDatabase.CreateAsset(tex, SavePath1);
        AssetDatabase.SaveAssets();

        Regex rgAlpha = new Regex("(?<=alpha:)[\\s\\S]*(?=index:)");
        Regex rgIndex = new Regex("(?<=index:)[\\s\\S]*\\]");

        Match mAlpha = rgAlpha.Match(source.text);
        string strAlpha = mAlpha.Groups[0].Value;
        Match mIndex = rgIndex.Match(source.text);
        string strIndex = mIndex.Groups[0].Value;

        if (strAlpha.Length == 0 || strIndex.Length == 0) {
            ShowNotification(new GUIContent("Wrong file format"));
            return;
        }

        // Extra +1 for rho and gamma constants
        Texture2D tex2 = new Texture2D(rows + 1, 2, TextureFormat.RFloat, false);

        MatchCollection alphaF = rgFloats.Matches(strAlpha);
        MatchCollection indexF = rgFloats.Matches(strIndex);

        for (i = 0; i < rows; i++) {
            float alpha = float.Parse(alphaF[i].Captures[0].Value);
            float index = float.Parse(indexF[i].Captures[0].Value);
            tex2.SetPixel(i + 1, 0, new Color(index, 0.0f, 0.0f, 0.0f));
            tex2.SetPixel(i + 1, 1, new Color(alpha, 0.0f, 0.0f, 0.0f));
        }

        Regex rgRho = new Regex("(?<=rho: )[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?");
        Match mRho = rgRho.Match(source.text);
        string strRho = mRho.Groups[0].Value;

        Regex rgGamma = new Regex("(?<=gamma: )[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?");
        Match mGamma = rgGamma.Match(source.text);
        string strGamma = mGamma.Groups[0].Value;

        tex2.SetPixel(0, 0, new Color(float.Parse(strRho), 0.0f, 0.0f, 0.0f));
        tex2.SetPixel(0, 1, new Color(float.Parse(strGamma), 0.0f, 0.0f, 0.0f));

        AssetDatabase.CreateAsset(tex2, SavePath2);
        AssetDatabase.SaveAssets();

        ShowNotification(new GUIContent("Done"));
        }
    }

#endif