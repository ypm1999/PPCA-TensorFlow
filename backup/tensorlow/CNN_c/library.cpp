#include "library.h"
#include <cstdio>
#include <iostream>
#include <cstring>
using std::cout;
using std::endl;

unsigned int num, n, m, fn, fm, fin, fout, nn, mm;
float *img, *flt, *result, *grad;
int *maxpos;

inline unsigned int imgpos(const unsigned int &i, const unsigned int &j, const unsigned int &p, const unsigned int &q){
    return q + fin * (p + m * (j + i * n));
}

inline unsigned int fltpos(const unsigned int &i, const unsigned int &j, const unsigned int &p, const unsigned int &q){
    return q + fout * (p + fin * (j + i * fm));
}

inline unsigned int respos(const unsigned int &i, const unsigned int &j, const unsigned int &p, const unsigned int &q){
    return q + fout * (p + m * (j + i * n));
}


inline unsigned int polpos(const unsigned int &i, const unsigned int &j, const unsigned int &p, const unsigned int &q){
    return q + fout * (p + mm * (j + i * nn));
}


extern "C"
void run(float *_img, float *_flt, float *_result,
          int _num, int _n, int _m, int _fn, int _fm, int _fin, int _fout) {
    img = _img;
    flt = _flt;
    result = _result;
    num = (unsigned int)_num;
    n = (unsigned int)_n;
    m = (unsigned int)_m;
    fn = (unsigned int)_fn;
    fm = (unsigned int)_fm;
    fin = (unsigned int)_fin;
    fout = (unsigned int)_fout;

}

extern "C"
void max_pool(float *_img, float *_result, int *_maxpos,
              int _num, int _n, int _m, int _sn, int _sm, int _fin){
    img = _img;
    result = _result;
    maxpos = _maxpos;
    num = (unsigned int)_num;
    n = (unsigned int)_n;
    m = (unsigned int)_m;
    fin = (unsigned int)_fin;
    nn = n / _sn;
    mm = m / _sm;
    memset(result, -0x80, num * nn * mm * fin);

    unsigned int l = 0, r = num;
    for(unsigned int now = l; now < r; now++){
        for(unsigned int i = 0, idi = 0; i < n; idi = ++i / _sn){
            for(unsigned int j = 0, idj = 0; j < m; idj = ++j / _sm){
                float *image = img + imgpos(now, i, j, 0u);
                float *res = result + polpos(now, idi, idj, 0u);
                int *pos = maxpos + imgpos(now, i, j, 0u);
                for(unsigned int k = 0; k < fin; k++){
                    if(image[k] > res[k]){
                        res[k] = image[k];
                        pos[k] = i * n + j;
                    }
                }
            }
        }
    }
}

extern "C"
void max_pool_backup(int *_maxpos, float *_grad, float *_result,
              int _num, int _n, int _m, int _sn, int _sm, int _fin){

    maxpos = _maxpos;
    grad = _grad;
    result = _result;
    num = (unsigned int)_num;
    n = (unsigned int)_n;
    m = (unsigned int)_m;
    fin = (unsigned int)_fin;
    nn = n / _sn;
    mm = m / _sm;
    //memset(result, -0x80, num * nn * mm * fin);
    unsigned int l = 0, r = num;
    for(unsigned int now = l; now < r; now++){
        for(unsigned int i = 0, idi = 0; i < n; idi = ++i / _sn){
            for(unsigned int j = 0, idj = 0; j < m; idj = ++j / _sm){
                float *res = result + polpos(now, idi, idj, 0u);
                float *gra = grad + polpos(now, idi, idj, 0u);
                int *pos = maxpos + imgpos(now, i, j, 0u);
                register int tmp = i * n + j;
                for(unsigned int k = 0; k < fin; k++){
                    if(tmp == pos[k])
                        res[k] = gra[k];
                    else
                        res[k] = 0;
                }
            }
        }
    }
}

//int main(){
//    float a[10];
//    memset(a, 0x80, sizeof(a));
//    cout << a[0] << endl;
//    return 0;
//}