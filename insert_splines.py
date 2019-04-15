import pandas as pd

from sklearn.pipeline import Pipeline

from basis_expansions.basis_expansions import (
    Polynomial, LinearSpline)

from regression_tools.dftransformers import (
    ColumnSelector, Identity, FeatureUnion, MapFeature, Intercept)

from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier

def insert_splines(X):
    
    PMI_fit = Pipeline([
        ('PMI', ColumnSelector(name='PMI')),
        ('PMI_spline', LinearSpline(knots=[44,45,47,48]))
    ])

    CONS_SENT_fit = Pipeline([
        ('CONS_SENT', ColumnSelector(name='CONS_SENT')),
        ('CONS_SENT_spline', LinearSpline(knots=[64,68,72,75]))
    ])

    INT_RATE_fit = Pipeline([
        ('INT_RATE', ColumnSelector(name='INT_RATE')),
        ('INT_RATE_spline', LinearSpline(knots=[8.75]))
    ])

    US_NHOME_SALES_fit = Pipeline([
        ('US_NHOME_SALES', ColumnSelector(name='US_NHOME_SALES')),
        ('US_NHOME_SALES_spline', LinearSpline(knots=[450,475,500]))
    ])

    THREE_YRT_fit = Pipeline([
        ('3YRT', ColumnSelector(name='3YRT')),
        ('3YRT_spline', LinearSpline(knots=[11.5]))
    ])

    Spread_fit = Pipeline([
        ('Spread', ColumnSelector(name='Spread')),
        ('Spread_spline', LinearSpline(knots=[-0.1]))
    ])

    CONS_SENT_1m_shift_fit = Pipeline([
        ('CONS_SENT_1m_shift', ColumnSelector(name='CONS_SENT_1m_shift')),
        ('CONS_SENT_1m_shift_spline', LinearSpline(knots=[-5]))
    ])


    CAP_UTIL_1m_shift_fit = Pipeline([
        ('CAP_UTIL_1m_shift', ColumnSelector(name='CAP_UTIL_1m_shift')),
        ('CAP_UTIL_1m_shift_spline', LinearSpline(knots=[-0.5,-0.35,-0.25]))
    ])

    THREE_YRT_1m_shift_fit = Pipeline([
        ('3YRT_1m_shift', ColumnSelector(name='3 YRT_1m_shift')),
        ('3YRT_1m_shift_spline', LinearSpline(knots=[-0.48,-0.46]))
    ])

    THREE_MTREAS_YIELD_1m_shift_fit = Pipeline([
        ('3 Month Treasury Yield (Bond Equivalent Basis)_1m_shift', ColumnSelector(name='3 Month Treasury Yield (Bond Equivalent Basis)_1m_shift')),
        ('3 Month Treasury Yield (Bond Equivalent Basis)_1m_shift_spline', LinearSpline(knots=[-0.625,-0.45]))
    ])

    PMI_3m_shift_fit = Pipeline([
        ('PMI_3m_shift', ColumnSelector(name='PMI_3m_shift')),
        ('PMI_3m_shift_spline', LinearSpline(knots=[-6.5,-5.5,-5.25,-3]))
    ])

    UNR_3m_shift_fit = Pipeline([
        ('UNR_3m_shift', ColumnSelector(name='UNR_3m_shift')),
        ('UNR_3m_shift_spline', LinearSpline(knots=[0.1,0.4]))
    ])

    YUNR_3m_shift_fit = Pipeline([
        ('YUNR_3M_SHIFT', ColumnSelector(name='YUNR_3m_shift')),
        ('YUNR_3M_SHIFT_spline', LinearSpline(knots=[0.5,1]))
    ])

    CONS_SENT_3m_shift_fit = Pipeline([
        ('CONS_SENT_3m_shift', ColumnSelector(name='CONS_SENT_3m_shift')),
        ('CONS_SENT_3m_shift_spline', LinearSpline(knots=[-7,-4.5]))
    ])

    CAP_UTIL_3m_shift_fit = Pipeline([
        ('CAP_UTIL_3m_shift', ColumnSelector(name='CAP_UTIL_3m_shift')),
        ('CAP_UTIL_3m_shift_spline', LinearSpline(knots=[-1.6,-1.5,-1,-0.9]))
    ])

    THREE_MTREAS_YIELD_3m_shift_fit = Pipeline([
        ('3 Month Treasury Yield (Bond Equivalent Basis)_3m_shift', ColumnSelector(name='3 Month Treasury Yield (Bond Equivalent Basis)_3m_shift')),
        ('3 Month Treasury Yield (Bond Equivalent Basis)_3m_shift_spline', LinearSpline(knots=[-1.4]))
    ])

    UNR_12m_shift_fit = Pipeline([
        ('UNR_12m_shift', ColumnSelector(name='UNR_12m_shift')),
        ('UNR_12m_shift_spline', LinearSpline(knots=[2.5,3]))
    ])

    CONS_SENT_12m_shift_fit = Pipeline([
        ('CONS_SENT_12m_shift', ColumnSelector(name='CONS_SENT_12m_shift')),
        ('CONS_SENT_12m_shift_spline', LinearSpline(knots=[-15,-12,-11]))
    ])

    HOUS_PERMS_12m_shift_fit = Pipeline([
        ('HOUS_PERMS_12m_shift', ColumnSelector(name='HOUS_PERMS_12m_shift')),
        ('HOUS_PERMS_12m_shift_spline', LinearSpline(knots=[-375,-300,-200]))
    ])

    HOUS_STARTS_12m_shift_fit = Pipeline([
        ('HOUS_STARTS_12m_shift', ColumnSelector(name='HOUS_STARTS_12m_shift')),
        ('HOUS_STARTS_12m_shift_spline', LinearSpline(knots=[-375,-300]))
    ])

    THREE_YRT_12m_shift_fit = Pipeline([
        ('THREE_YRT_12m_shift', ColumnSelector(name='3YRT_12m_shift')),
        ('THREE_YRT_12m_shift_spline', LinearSpline(knots=[1.75]))
    ])

    TEN_YRT_YIELD_12m_shift_fit = Pipeline([
        ('TEN_YRT_YIELD_12m_shift', ColumnSelector(name='10 Year Treasury Yield_12m_shift')),
        ('10 Year Treasury Yield_12m_shift_spline', LinearSpline(knots=[1.2,1.6]))
    ])

    Spread_12m_shift_fit = Pipeline([
        ('Spread_12m_shift', ColumnSelector(name='Spread_12m_shift')),
        ('Spread_12m_shift_spline', LinearSpline(knots=[1,1.1]))
    ])

    PPI_1m_shift_fit = Pipeline([
        ('PPI_1m_shift', ColumnSelector(name='PPI_1m_shift')),
        ('PPI_1m_shift_spline', LinearSpline(knots=[1.3]))
    ])

    CPI_1m_shift_fit = Pipeline([
        ('CPI_1m_shift', ColumnSelector(name='CPI_1m_shift')),
        ('CPI_1m_shift_spline', LinearSpline(knots=[1.8]))
    ])

    PPI_3m_shift_fit = Pipeline([
        ('PPI_3m_shift', ColumnSelector(name='PPI_3m_shift')),
        ('PPI_3m_shift_spline', LinearSpline(knots=[2.75,3,3.25,3.35]))
    ])

    CPI_12m_shift_fit = Pipeline([
        ('CPI_12m_shift', ColumnSelector(name='CPI_12m_shift')),
        ('CPI_12m_shift_spline', LinearSpline(knots=[5.5]))
    ])

    PPI_12m_shift_fit = Pipeline([
        ('PPI_12m_shift', ColumnSelector(name='PPI_12m_shift')),
        ('PPI_12m_shift_spline', LinearSpline(knots=[14]))
    ])


    #union features together

    feature_pipeline = FeatureUnion([
        ('intercept', Intercept()),
        ('PMI', PMI_fit),
        ('CONS_SENT', CONS_SENT_fit),
        ('INT_RATE', INT_RATE_fit),
        ('US_NHOME_SALES', US_NHOME_SALES_fit),
        ('THREE_YRT', THREE_YRT_fit),
        ('Spread', Spread_fit),
        ('CONS_SENT_1m_shift', CONS_SENT_1m_shift_fit),
        ('CAP_UTIL_1m_shift', CAP_UTIL_1m_shift_fit),
        ('THREE_MTREAS_YIELD_1m_shift', THREE_MTREAS_YIELD_1m_shift_fit),
        ('PMI_3m_shift', PMI_3m_shift_fit),
        ('UNR_3m_shift', UNR_3m_shift_fit),
        ('YUNR_3M_SHIFT', YUNR_3m_shift_fit),
        ('CONS_SENT_3m_shift', CONS_SENT_3m_shift_fit),
        ('CAP_UTIL_3m_shift', CAP_UTIL_3m_shift_fit),
        ('3 Month Treasury Yield_3m_shift', THREE_MTREAS_YIELD_3m_shift_fit),
        ('UNR_12m_shift', UNR_12m_shift_fit),
        ('CONS_SENT_12m_shift', CONS_SENT_12m_shift_fit),
        ('HOUS_PERMS_12m_shift', HOUS_PERMS_12m_shift_fit),
        ('HOUS_STARTS_12m_shift', HOUS_STARTS_12m_shift_fit), 
        ('THREE_YRT_12m_shift', THREE_YRT_12m_shift_fit),
        ('TEN_YRT_YIELD_12m_shift', TEN_YRT_YIELD_12m_shift_fit),
        ('Spread_12m_shift', Spread_12m_shift_fit),
        ('PPI_1m_shift', PPI_1m_shift_fit),
        ('CPI_1m_shift', CPI_1m_shift_fit),
        ('PPI_3m_shift', PPI_3m_shift_fit),
        ('CPI_12m_shift', CPI_12m_shift_fit),
        ('PPI_12m_shift', PPI_12m_shift_fit)
        ])


    feature_pipeline.fit(X)
    features = feature_pipeline.transform(X)
    
    #dropping columns from OG dataset that were splined

    splined_cols = ['PMI','CONS_SENT','INT_RATE','US_NHOME_SALES','3YRT','Spread','CONS_SENT_1m_shift','CAP_UTIL_1m_shift',
                    '3YRT_1m_shift', '3 Month Treasury Yield (Bond Equivalent Basis)_1m_shift', 'PMI_3m_shift', 
                    'UNR_3m_shift', 'YUNR_3m_shift', 'CONS_SENT_3m_shift', 'CAP_UTIL_3m_shift',
                    '3 Month Treasury Yield (Bond Equivalent Basis)_3m_shift',
                    'UNR_12m_shift', 'CONS_SENT_12m_shift', 'HOUS_PERMS_12m_shift','HOUS_STARTS_12m_shift',
                    '3YRT_12m_shift','10 Year Treasury Yield_12m_shift','Spread_12m_shift','PPI_1m_shift',
                    'CPI_3m_shift','PPI_3m_shift','CPI_12m_shift','PPI_12m_shift']

    X = X.drop(columns = splined_cols)
    
    X = X.join(features, how='outer')
    
    return X
