using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Warp.Sociology;

namespace M.Controls.Sociology
{
    /// <summary>
    /// Interaction logic for SpeciesView.xaml
    /// </summary>
    public partial class SpeciesView : UserControl
    {
        public Species Species
        {
            get { return (Species)GetValue(SpeciesProperty); }
            set { SetValue(SpeciesProperty, value); }
        }
        public static readonly DependencyProperty SpeciesProperty = DependencyProperty.Register("Species", typeof(Species), typeof(SpeciesView), new PropertyMetadata(null, (sender, args) => ((SpeciesView)sender).DataContext = args.NewValue));
        

        public SpeciesView()
        {
            InitializeComponent();
        }
    }
}
