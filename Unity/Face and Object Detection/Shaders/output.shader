Shader "FaceAndObjectDetect/output"
{
    Properties
    {
    	_TexCam ("High Res Camera", 2D) = "black" {}
        _TexClass ("Classify Output", 2D) = "black" {}
        _TexSprite ("Sprite sheet", 2D) = "black" {}
        [Toggle]_Box ("Bounding Box", Float) = 1
        _Color ("Color", Color) = (0.,1.,0.,1)
        [Toggle]_Emojis ("Emojis", Float) = 1
        _Size ("Size", Float) = 0.02
    }
    SubShader
    {
        Cull Off
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma fragmentoption ARB_precision_hint_fastest
            #pragma target 5.0

            #include "UnityCG.cginc"
            #define _SCALE_TIME 0.75

            Texture2D<float4> _TexCam;
            Texture2D<float4> _TexClass;
            Texture2D<float4> _TexSprite;
            float4 _TexCam_TexelSize;
            float4 _TexClass_TexelSize;
            float4 _TexSprite_TexelSize;
            float4 _Color;
            float _Box;
            float _Emojis;
            float _Size;

            //RWStructuredBuffer<float4> buffer : register(u1);

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            inline float box(float2 uv) {
                uv = step(0.95, uv);
                return max(uv.x, uv.y);
            }

            // max frames per column
            static const uint fmax[7] = { 12, 72, 10, 20, 60, 41, 10 };

            float4 sprite(float2 uv, uint i, float h) {
                uv.x = uv.x * 64 + i * _TexSprite_TexelSize.z / 7.;
                uv.y = uv.y * 32 + (round(_Time.y * 40 + 80 * h) % fmax[i]) * _TexSprite_TexelSize.w / 72.;
                float4 col = _TexSprite.Load(uint3(uv, 0));
                return col;
            }

            ///  2 out, 2 in...
            float2 hash22(float2 p)
            {
                float3 p3 = frac(float3(p.xyx) * float3(.1031, .1030, .0973));
                p3 += dot(p3, p3.yzx+19.19);
                return frac((p3.xx+p3.yz)*p3.zy);
            }

            float4 frag (v2f ps) : SV_Target
            {
                float4 col = _TexCam.Load(uint3(ps.uv * _TexCam_TexelSize.zw, 0));
                float2 size = _Size.xx;
                size.y = size.y / _TexCam_TexelSize.w * _TexCam_TexelSize.z;
                
                // The edges of the classified output has a lot of false 
                // positives
                for (uint x = 0; x < (uint)(_TexClass_TexelSize.z); x++) {
                    for (uint y = 0; y < (uint)(_TexClass_TexelSize.w); y++) {
                        bool isFace = _TexClass.Load(uint3(x, y, 0)).r > 0.5 ? true : false;
                        if (isFace && _Box) {
                            // muv.y = 1 - muv.y; to flip the output back again
                            // cause every stupid program wants to start 0,0 
                            // at different fucking places
                            float2 muv = float2(x, y) / _TexClass_TexelSize.zw;
                            muv.y = 1 - muv.y;
                            float2 dist = ps.uv - muv;
                            [flatten]
                            if (abs(dist.x) < size.x * 4. && abs(dist.y) < size.y * 4.) {
                                col += box(abs(dist) / size / 4.) * _Color;
                            }
                        }
                        if (isFace && _Emojis) {
                            [unroll]
                            for (uint i = 0; i < 7; i++) {
                                float2 h = hash22(x * y * (i + 1));
                                float2 muv = float2(x, y) / _TexClass_TexelSize.zw;
                                muv.y = 1 - muv.y;
                                muv += (h - 0.5) / 3.0;
                                float2 dist = abs(ps.uv - muv);
                                if (dist.x < size.x && dist.y < size.y) {
                                    float4 emojis = sprite((ps.uv - muv + size) / size.x * 0.25, i, h.y*h.x);
                                    col.rgb = lerp(emojis.rgb, col.rgb, 1.0 - emojis.a);
                                }
                            }
                        }
                    }
                }
                return min(col, 1.5);
            }
            ENDCG
        }
    }
}
