from setuptools import setup, find_packages

setup(
   name='khan',
   version='1.0',
   description='Spectrogram-UMAP-Based Temporal Link Embedding',
   author='Jea Kwon',
   author_email='onlytojay@gmail.com',
   packages=find_packages(),  # would be the same as name
   install_requires=['numpy','pandas','scipy', 'scikit-learn', 'opentsne', 'umap-learn', 'PhenoGraph','tqdm'], 
   python_requires='>=3.7',
)
Footer
© 2023 GitH